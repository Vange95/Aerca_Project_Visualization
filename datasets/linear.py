# Some synthetic datasets with linear dynamics
import numpy as np
import os
import random


class Linear:
    def __init__(self, options):
        self.options = options
        self.data_dict = {}
        self.seed = options['seed']
        self.n = options['training_size'] + options['testing_size']
        self.t = options['T']
        self.mul = options.get('mul', 4.5)
        self.a = options['a'] if options['a'] is not None else self._generate_random_coefficients()
        self.adlength = int(max(1, options.get('adlength', 60)))
        self.adtype = options.get('adtype', 'spike')
        self.fault_id = options.get('fault_id')
        self.data_dir = options['data_dir']
        self.dependent_features = options.get('dependent_features', 0)

        self.fault_specs = {
            'equipment_spike': {'adtype': 'step_up', 'affected_vars': [1, 2]},
            'power_loss': {'adtype': 'drop_to_zero', 'affected_vars': [0, 1, 2, 3]},
            'sensor_drift': {'adtype': 'gradual_drift', 'affected_vars': [2]},
            'signal_loss': {'adtype': 'signal_zero', 'affected_vars': [3]},
            'actuator_stuck': {'adtype': 'stuck_value', 'affected_vars': [1]},
            'pressure_drop': {'adtype': 'step_down', 'affected_vars': [0, 1]},
        }
        if self.fault_id in self.fault_specs:
            self.adtype = self.fault_specs[self.fault_id]['adtype']

        self.supported_adtypes = [
            'spike', 'step', 'causal',
            'step_up', 'step_down', 'drop_to_zero', 'signal_zero',
            'gradual_drift', 'stuck_value',
        ]

        if self.adtype not in self.supported_adtypes:
            print(f"Warning: adtype '{self.adtype}' not supported. Using 'spike' instead.")
            self.adtype = 'spike'

    def _generate_random_coefficients(self):
        a = np.zeros((8,))
        for k in range(8):
            u_1 = np.random.uniform(0, 1)
            a[k] = np.random.uniform(-0.8, -0.2) if u_1 <= 0.5 else np.random.uniform(0.2, 0.8)
        return a

    def generate_example(self):
        if self.seed is not None:
            np.random.seed(self.seed)
            random.seed(self.seed)

        x_n_list = np.zeros((self.n, self.t, 4))
        x_ab_list = np.zeros((self.n, self.t, 4))
        label_list = np.zeros((self.n, self.t, 4))

        for i in range(self.n):
            eps = 0.4 * np.random.randn(self.t, 4)

            # 正常序列
            x = np.zeros(self.t)
            w = np.zeros(self.t)
            y = np.zeros(self.t)
            z = np.zeros(self.t)
            for j in range(1, self.t):
                x[j] = self.a[0] * x[j - 1] + eps[j, 0]
                w[j] = self.a[1] * w[j - 1] + self.a[2] * x[j - 1] + eps[j, 1]
                y[j] = self.a[3] * y[j - 1] + self.a[4] * w[j - 1] + eps[j, 2]
                z[j] = self.a[5] * z[j - 1] + self.a[6] * w[j - 1] + self.a[7] * y[j - 1] + eps[j, 3]

            x_n_list[i] = np.stack((x, w, y, z), axis=-1)

            # ====================== 生成异常（最终稳定版） ======================
            effective_adlength = min(self.adlength, max(1, int(self.t * 0.35)))
            start_low = max(0, int(0.2 * self.t))
            start_high = max(start_low + 1, int(0.8 * self.t) - effective_adlength)
            start = np.random.randint(start_low, start_high)
            t_p = np.arange(start, min(self.t, start + effective_adlength))
            if self.fault_id in self.fault_specs:
                feature_p = np.array(self.fault_specs[self.fault_id]['affected_vars'], dtype=int)
            else:
                num_features = np.random.randint(1, 4)
                feature_p = np.random.choice(4, size=num_features, replace=False)

            temp_label = np.zeros((self.t, 4))
            temp_label[np.ix_(t_p, feature_p)] = 1

            x_ab = x.copy()
            w_ab = w.copy()
            y_ab = y.copy()
            z_ab = z.copy()

            if self.adtype == 'spike':
                amp = self.mul * 2.0
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] += amp
                    elif f == 1:
                        w_ab[t_p] += amp
                    elif f == 2:
                        y_ab[t_p] += amp
                    elif f == 3:
                        z_ab[t_p] += amp

            elif self.adtype in ('step', 'step_up'):
                step_value = self.mul * 2.0
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] += step_value
                    elif f == 1:
                        w_ab[t_p] += step_value
                    elif f == 2:
                        y_ab[t_p] += step_value
                    elif f == 3:
                        z_ab[t_p] += step_value

            elif self.adtype == 'step_down':
                step_value = self.mul * 2.0
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] -= step_value
                    elif f == 1:
                        w_ab[t_p] -= step_value
                    elif f == 2:
                        y_ab[t_p] -= step_value
                    elif f == 3:
                        z_ab[t_p] -= step_value

            elif self.adtype in ('drop_to_zero', 'signal_zero'):
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] = 0.0
                    elif f == 1:
                        w_ab[t_p] = 0.0
                    elif f == 2:
                        y_ab[t_p] = 0.0
                    elif f == 3:
                        z_ab[t_p] = 0.0

            elif self.adtype == 'gradual_drift':
                drift = np.linspace(0.0, self.mul * 2.0, len(t_p))
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] += drift
                    elif f == 1:
                        w_ab[t_p] += drift
                    elif f == 2:
                        y_ab[t_p] += drift
                    elif f == 3:
                        z_ab[t_p] += drift

            elif self.adtype == 'stuck_value':
                hold_t = max(0, start - 1)
                for f in feature_p:
                    if f == 0:
                        x_ab[t_p] = x_ab[hold_t]
                    elif f == 1:
                        w_ab[t_p] = w_ab[hold_t]
                    elif f == 2:
                        y_ab[t_p] = y_ab[hold_t]
                    elif f == 3:
                        z_ab[t_p] = z_ab[hold_t]

            elif self.adtype == 'causal':
                b = self.a * 4.8
                for j in range(1, self.t):
                    if start <= j < start + self.adlength:
                        x_ab[j] = b[0] * x_ab[j - 1] + eps[j, 0]
                        w_ab[j] = b[1] * w_ab[j - 1] + b[2] * x_ab[j - 1] + eps[j, 1]
                        y_ab[j] = b[3] * y_ab[j - 1] + b[4] * w_ab[j - 1] + eps[j, 2]
                        z_ab[j] = b[5] * z_ab[j - 1] + b[6] * w_ab[j - 1] + b[7] * y_ab[j - 1] + eps[j, 3]
                    else:
                        x_ab[j] = self.a[0] * x_ab[j - 1] + eps[j, 0]
                        w_ab[j] = self.a[1] * w_ab[j - 1] + self.a[2] * x_ab[j - 1] + eps[j, 1]
                        y_ab[j] = self.a[3] * y_ab[j - 1] + self.a[4] * w_ab[j - 1] + eps[j, 2]
                        z_ab[j] = self.a[5] * z_ab[j - 1] + self.a[6] * w_ab[j - 1] + self.a[7] * y_ab[j - 1] + eps[
                            j, 3]

            x_ab_list[i] = np.stack((x_ab, w_ab, y_ab, z_ab), axis=-1)
            label_list[i] = temp_label

        self.data_dict = {
            'x_n_list': x_n_list,
            'x_ab_list': x_ab_list,
            'label_list': label_list,
            'a': self.a,
            'causal_struct': np.array([[1, 0, 0, 0], [1, 1, 0, 0], [0, 1, 1, 0], [0, 1, 1, 1]]),
            'causal_struct_value': np.array(
                [[self.a[0], 0, 0, 0], [self.a[2], self.a[1], 0, 0], [0, self.a[4], self.a[3], 0],
                 [0, self.a[6], self.a[7], self.a[5]]]),
            'signed_causal_struct': np.sign(np.array(
                [[self.a[0], 0, 0, 0], [self.a[2], self.a[1], 0, 0], [0, self.a[4], self.a[3], 0],
                 [0, self.a[6], self.a[7], self.a[5]]]))
        }

    def save_data(self):
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        np.save(os.path.join(self.data_dir, 'x_n_list.npy'), self.data_dict['x_n_list'])
        np.save(os.path.join(self.data_dir, 'x_ab_list.npy'), self.data_dict['x_ab_list'])
        np.save(os.path.join(self.data_dir, 'label_list.npy'), self.data_dict['label_list'])
        np.save(os.path.join(self.data_dir, 'a.npy'), self.data_dict['a'])
        np.save(os.path.join(self.data_dir, 'causal_struct.npy'), self.data_dict['causal_struct'])
        np.save(os.path.join(self.data_dir, 'causal_struct_value.npy'), self.data_dict['causal_struct_value'])
        np.save(os.path.join(self.data_dir, 'signed_causal_struct.npy'), self.data_dict['signed_causal_struct'])

    def load_data(self):
        self.data_dict['x_n_list'] = np.load(os.path.join(self.data_dir, 'x_n_list.npy'))
        self.data_dict['x_ab_list'] = np.load(os.path.join(self.data_dir, 'x_ab_list.npy'))
        self.data_dict['label_list'] = np.load(os.path.join(self.data_dir, 'label_list.npy'))
        self.data_dict['a'] = np.load(os.path.join(self.data_dir, 'a.npy'))
        self.data_dict['causal_struct'] = np.load(os.path.join(self.data_dir, 'causal_struct.npy'))
        self.data_dict['causal_struct_value'] = np.load(os.path.join(self.data_dir, 'causal_struct_value.npy'))
        self.data_dict['signed_causal_struct'] = np.load(os.path.join(self.data_dir, 'signed_causal_struct.npy'))
