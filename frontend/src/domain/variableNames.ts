const DATASET_VARIABLE_NAMES: Record<string, string[]> = {
  linear: [
    '上游输出 (var_0)',
    '中间压力 (var_1)',
    '下游响应 (var_2)',
    '末端反馈 (var_3)',
  ],
  nonlinear: [
    '负载输入 (var_0)',
    '耦合压力 (var_1)',
    '温度读数 (var_2)',
    '振荡响应 (var_3)',
    '执行反馈 (var_4)',
    '采集通道 (var_5)',
  ],
  swat: [
    '进水流量 (var_0)',
    '泵出口压力 (var_1)',
    '管路流量 (var_2)',
    '阀门开度 (var_3)',
    '水箱液位 (var_4)',
    'PLC通道A (var_5)',
    'PLC通道B (var_6)',
  ],
}

const CLOSED_LOOP_DATASETS = new Set(['linear', 'nonlinear', 'swat'])

const FAULT_VARIABLE_NAMES: Record<string, Record<string, string[]>> = {
  linear: {
    equipment_spike: [
      '上游设备输出 (var_0)',
      '异常升高输出 (var_1)',
      '下游联动响应 (var_2)',
      '末端反馈 (var_3)',
    ],
    power_loss: [
      '供电支路A (var_0)',
      '控制器供电 (var_1)',
      '传感器供电 (var_2)',
      '通信模块反馈 (var_3)',
    ],
    sensor_drift: [
      '工艺参考量 (var_0)',
      '过程联动量 (var_1)',
      '漂移传感器读数 (var_2)',
      '末端反馈 (var_3)',
    ],
    signal_loss: [
      '上游采集值 (var_0)',
      '中间过程值 (var_1)',
      '下游过程值 (var_2)',
      '通信/采集链路 (var_3)',
    ],
    actuator_stuck: [
      '控制指令 (var_0)',
      '执行器位置/压力 (var_1)',
      '过程响应 (var_2)',
      '反馈信号 (var_3)',
    ],
    pressure_drop: [
      '上游压力源 (var_0)',
      '管路压力 (var_1)',
      '下游响应 (var_2)',
      '末端反馈 (var_3)',
    ],
  },
  nonlinear: {
    nonlinear_load_surge: [
      '负载输入 (var_0)',
      '耦合压力 (var_1)',
      '温度响应 (var_2)',
      '振荡响应 (var_3)',
      '执行反馈 (var_4)',
      '采集通道 (var_5)',
    ],
    nonlinear_sensor_drift: [
      '负载输入 (var_0)',
      '耦合压力 (var_1)',
      '漂移温度传感器 (var_2)',
      '振荡响应 (var_3)',
      '执行反馈 (var_4)',
      '采集通道 (var_5)',
    ],
    nonlinear_instability: [
      '负载输入 (var_0)',
      '失稳耦合压力 (var_1)',
      '温度读数 (var_2)',
      '异常振荡响应 (var_3)',
      '执行反馈 (var_4)',
      '采集通道 (var_5)',
    ],
    nonlinear_signal_loss: [
      '负载输入 (var_0)',
      '耦合压力 (var_1)',
      '温度读数 (var_2)',
      '振荡响应 (var_3)',
      '执行反馈 (var_4)',
      '丢失采集通道 (var_5)',
    ],
  },
  swat: {
    swat_pump_trip: [
      '水泵进水流量 (var_0)',
      '水泵出口压力 (var_1)',
      '管路流量 (var_2)',
      '阀门开度 (var_3)',
      '水箱液位 (var_4)',
      'PLC通道A (var_5)',
      'PLC通道B (var_6)',
    ],
    swat_valve_stuck: [
      '进水流量 (var_0)',
      '泵出口压力 (var_1)',
      '管路流量 (var_2)',
      '卡滞阀门开度 (var_3)',
      '水箱液位 (var_4)',
      'PLC通道A (var_5)',
      'PLC通道B (var_6)',
    ],
    swat_level_sensor_drift: [
      '进水流量 (var_0)',
      '泵出口压力 (var_1)',
      '管路流量 (var_2)',
      '阀门开度 (var_3)',
      '漂移液位传感器 (var_4)',
      'PLC通道A (var_5)',
      'PLC通道B (var_6)',
    ],
    swat_quality_sensor_fault: [
      '进水流量 (var_0)',
      '泵出口压力 (var_1)',
      '水质/压力读数 (var_2)',
      '阀门开度 (var_3)',
      '联动液位读数 (var_4)',
      'PLC通道A (var_5)',
      'PLC通道B (var_6)',
    ],
  },
}

function fallbackVariableName(dataset: string, index: number) {
  if (dataset === 'swat') return `水处理测点 var_${index}`
  if (dataset === 'msds') return `服务指标 metric_${index}`
  if (dataset === 'nonlinear') return `耦合变量 var_${index}`
  return `var_${index}`
}

export function getVariableNames(dataset: string, count: number, faultId?: string | null): string[] {
  if (!CLOSED_LOOP_DATASETS.has(dataset)) {
    return Array.from({ length: count }, (_, i) => fallbackVariableName(dataset, i))
  }

  if (dataset === 'lotka_volterra') {
    const p = Math.floor(count / 2)
    return [
      ...Array.from({ length: p }, (_, i) => `Prey_${i}`),
      ...Array.from({ length: count - p }, (_, i) => `Predator_${i}`),
    ]
  }

  if (dataset === 'lorenz96') {
    return Array.from({ length: count }, (_, i) => `X_${i}`)
  }

  const names = (faultId ? FAULT_VARIABLE_NAMES[dataset]?.[faultId] : undefined) ?? DATASET_VARIABLE_NAMES[dataset] ?? []
  return Array.from({ length: count }, (_, i) => names[i] ?? fallbackVariableName(dataset, i))
}

export function formatVariableList(dataset: string, vars: number[], count?: number, faultId?: string | null) {
  const names = getVariableNames(dataset, count ?? Math.max(0, ...vars) + 1, faultId)
  return vars.map((v) => names[v] ?? fallbackVariableName(dataset, v)).join(' + ')
}
