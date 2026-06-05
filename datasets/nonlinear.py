import numpy as np
import random
import igraph as ig
import os
from tqdm import tqdm


class Nonlinear:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.m = options['m']
        self.num_vars = options['num_vars']
        self.data_dir = options['data_dir']
        self.mul = options['mul']
        self.adlength = options['adlength']
        self.adtype = options['adtype']
        self.fault_id = options.get('fault_id')
        self.noise_scale = options['noise_scale']
        self.dependent_features = options['dependent_features']
        self.fault_specs = {
            'nonlinear_load_surge': {
                'adtype': 'step_up',
                'affected_vars': [0, 1, 2],
                'duration': 10,
            },
            'nonlinear_sensor_drift': {
                'adtype': 'gradual_drift',
                'affected_vars': [2],
                'duration': 18,
            },
            'nonlinear_instability': {
                'adtype': 'oscillation',
                'affected_vars': [1, 3, 4],
                'duration': 14,
            },
            'nonlinear_signal_loss': {
                'adtype': 'signal_zero',
                'affected_vars': [5],
                'duration': 10,
            },
        }
        if self.fault_id in self.fault_specs:
            self.adtype = self.fault_specs[self.fault_id]['adtype']
        self.generate_er_graph()

    def generate_er_graph(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
        # Generate a directed Erdös-Renyi graph and store its transposed adjacency matrix.
        ig.set_random_number_generator(random.Random(self.seed))
        G_und = ig.Graph.Erdos_Renyi(n=self.num_vars, m=self.m, directed=True, loops=False)
        self.data_dict['causal_struct'] = np.array(G_und.get_adjacency().data).T
        self.data_dict['signed_causal_struct'] = None

    def generate_example(self):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        x_n_list = []
        x_ab_list = []
        eps_n_list = []
        eps_ab_list = []
        label_list = []

        coefficients = np.random.uniform(low=0.1, high=2.0, size=(self.num_vars, self.num_vars, 5))

        for i in tqdm(range(self.n)):
            # Generate noise based on dependency flag.
            if self.dependent_features == 1:
                # Generate features with specified covariance
                # Define a covariance matrix manually
                covariance_matrix = np.array([
                    [1.0, 0.8, 0.6, 0.4, 0.2, 0.1],
                    [0.8, 1.0, 0.7, 0.5, 0.3, 0.2],
                    [0.6, 0.7, 1.0, 0.6, 0.4, 0.3],
                    [0.4, 0.5, 0.6, 1.0, 0.5, 0.4],
                    [0.2, 0.3, 0.4, 0.5, 1.0, 0.6],
                    [0.1, 0.2, 0.3, 0.4, 0.6, 1.0]
                ])
                mean = np.zeros(self.num_vars)
                eps = self.noise_scale * np.random.multivariate_normal(mean, covariance_matrix, size=self.t)
            else:
                eps = self.noise_scale * np.random.randn(self.t, self.num_vars)

            # Make separate copies for normal and anomalous series.
            eps_normal = eps.copy()
            eps_anom = eps.copy()

            # Initialize time series arrays with random initial values for the first 5 time steps.
            x = np.zeros((self.t, self.num_vars))
            x[:5] = np.random.randn(5, self.num_vars)
            x_ab = np.zeros((self.t, self.num_vars))
            x_ab[:5] = x[:5].copy()

            A_list = [self.data_dict['causal_struct'] * coefficients[:, :, lag] for lag in range(5)]

            # Set up anomaly parameters. Named fault scenarios use fixed variables
            # and duration so offline samples match the realtime root-cause demo.
            fault_spec = self.fault_specs.get(self.fault_id)
            duration = int(fault_spec['duration']) if fault_spec else int(self.adlength)
            duration = min(max(1, duration), max(1, int(self.t * 0.35)))
            start_low = max(5, int(0.2 * self.t))
            start_high = max(start_low + 1, int(0.8 * self.t) - duration)
            start = np.random.randint(start_low, start_high)
            t_p = np.arange(start, min(self.t, start + duration))
            t_p_set = set(int(t) for t in t_p)  # Use a set for O(1) membership checking.
            if fault_spec:
                feature_p = np.array(
                    [v for v in fault_spec['affected_vars'] if v < self.num_vars],
                    dtype=int,
                )
                if len(feature_p) == 0:
                    feature_p = np.array([0], dtype=int)
            else:
                feature_count = np.random.randint(1, min(10, self.num_vars) + 1)
                feature_p = np.random.permutation(np.arange(self.num_vars))[:feature_count]
            ab = np.zeros(self.num_vars)
            ab[feature_p] += self.mul
            temp_label = np.zeros((self.t, self.num_vars))
            temp_label[t_p, feature_p] = 1

            # Generate the normal time series x using the vectorized inner loop.
            for t in range(5, self.t):
                # Sum contributions from the previous 5 time steps.
                contributions = sum(
                    A_list[lag].dot(np.cos(x[t - lag - 1, :] + 1)) for lag in range(5)
                )
                x[t, :] = contributions + eps_normal[t, :]

            # Generate the anomalous time series x_ab.
            for t in range(5, self.t):
                # For anomaly time steps, update the noise with the anomaly effect.
                if t in t_p_set:
                    anomaly_progress = (t - int(t_p[0]) + 1) / max(1, len(t_p))
                    if self.adtype in ('non_causal', 'step_up'):
                        eps_anom[t, :] += ab
                    elif self.adtype == 'step_down':
                        eps_anom[t, :] -= ab
                    elif self.adtype == 'gradual_drift':
                        eps_anom[t, feature_p] += self.mul * anomaly_progress
                    elif self.adtype == 'oscillation':
                        eps_anom[t, feature_p] += self.mul * 0.7 * np.sin(
                            (t - int(t_p[0]) + 1) * 1.35 + np.arange(len(feature_p))
                        )
                    elif self.adtype in ('signal_zero', 'drop_to_zero', 'stuck_value'):
                        pass
                    elif self.adtype == 'causal':
                        raise NotImplementedError("Causal anomaly not implemented for this dataset.")
                    else:
                        raise NotImplementedError(
                            "Invalid adtype. Expected a nonlinear fault pattern or 'non_causal'."
                        )
                contributions_ab = sum(
                    A_list[lag].dot(np.cos(x_ab[t - lag - 1, :] + 1)) for lag in range(5)
                )
                x_ab[t, :] = contributions_ab + eps_anom[t, :]
                if t in t_p_set:
                    if self.adtype in ('signal_zero', 'drop_to_zero'):
                        x_ab[t, feature_p] = 0.0
                    elif self.adtype == 'stuck_value':
                        x_ab[t, feature_p] = x_ab[max(0, int(t_p[0]) - 1), feature_p]

            x_n_list.append(x)
            eps_n_list.append(eps_normal)
            x_ab_list.append(x_ab)
            eps_ab_list.append(eps_anom)
            label_list.append(temp_label)

        # Save the generated lists into the data dictionary (done once after the loop for efficiency).
        self.data_dict['x_n_list'] = np.array(x_n_list)
        self.data_dict['x_ab_list'] = np.array(x_ab_list)
        self.data_dict['eps_n_list'] = np.array(eps_n_list)
        self.data_dict['eps_ab_list'] = np.array(eps_ab_list)
        self.data_dict['label_list'] = np.array(label_list)

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'eps_n_list'), self.data_dict['eps_n_list'])
        np.save(os.path.join(self.data_dir, 'eps_ab_list'), self.data_dict['eps_ab_list'])
        np.save(os.path.join(self.data_dir, 'causal_struct'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'label_list'), self.data_dict['label_list'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['eps_n_list'] = np.load(os.path.join(self.data_dir, 'eps_n_list.npy'))
        self.data_dict['eps_ab_list'] = np.load(os.path.join(self.data_dir, 'eps_ab_list.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['signed_causal_struct'] = None
