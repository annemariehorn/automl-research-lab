
/** NYC Taxi LangGraph workflow configuration. */

export type SimpleLoopNodeId =
  | 'load_data'
  | 'prepare_data'
  | 'hypothesis_generator'
  | 'memory_writer'
  | 'context_builder'
  | 'train_model'
  | 'evaluate_model'
  | 'final_test'

export type SimpleLoopNodeKind = 'file' | 'code' | 'code_llm' | 'store'

export interface SimpleLoopNode {
  id: SimpleLoopNodeId
  title: string
  role: string
  kind: SimpleLoopNodeKind
  service: string
}

export interface SimpleLoopEdge {
  id: string
  source: SimpleLoopNodeId
  target: SimpleLoopNodeId
  kind: 'control' | 'data'
  label?: string
  sourceHandle?: string
  targetHandle?: string
}

export const SIMPLE_LOOP_NODES: SimpleLoopNode[] = [
  {
    id: 'load_data',
    title: 'Load NYC Taxi Data',
    role: 'Validate cleaned Airflow Parquet output',
    kind: 'file',
    service: 'data ingestion',
  },
  {
    id: 'prepare_data',
    title: 'Prepare Data',
    role: 'Create fixed splits and training-median target',
    kind: 'code',
    service: 'data preprocessing',
  },
  {
    id: 'hypothesis_generator',
    title: 'Hypothesis Generator',
    role: 'Propose an untried model from validation feedback',
    kind: 'code',
    service: 'experiment planning',
  },
  {
    id: 'memory_writer',
    title: 'Memory Writer',
    role: 'File configuration, validation results and judge feedback',
    kind: 'store',
    service: 'experiment memory',
  },
  {
    id: 'context_builder',
    title: 'Context Builder',
    role: 'Summarize evidence for the next hypothesis',
    kind: 'code',
    service: 'research context',
  },
  {
    id: 'train_model',
    title: 'Train Model',
    role: 'Fit machine learning models',
    kind: 'code',
    service: 'model training',
  },
  {
    id: 'evaluate_model',
    title: 'Evaluate Model',
    role: 'Compare validation performance and decide whether to retry',
    kind: 'code',
    service: 'model evaluation',
  },
  {
    id: 'final_test',
    title: 'Final Test',
    role: 'Evaluate selected model on held-out test data',
    kind: 'code',
    service: 'final evaluation',
  },
]

export const SIMPLE_LOOP_EDGES: SimpleLoopEdge[] = [
  {
    id: 'e-load-prepare',
    source: 'load_data',
    target: 'prepare_data',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-prepare-hypothesis',
    source: 'prepare_data',
    target: 'hypothesis_generator',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-hypothesis_generator-train_model',
    source: 'hypothesis_generator',
    target: 'train_model',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-memory_writer-context_builder',
    label: 'retry',
    source: 'memory_writer',
    target: 'context_builder',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-context_builder-hypothesis_generator',
    source: 'context_builder',
    target: 'hypothesis_generator',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-train-evaluate',
    source: 'train_model',
    target: 'evaluate_model',
    kind: 'control',
    sourceHandle: 'right-out',
    targetHandle: 'left-in',
  },
  {
    id: 'e-evaluate-memory',
    source: 'evaluate_model',
    target: 'memory_writer',
    kind: 'control',
    sourceHandle: 'top-out',
    targetHandle: 'top-in',
  },
  {
    id: 'e-memory-final',
    source: 'memory_writer',
    target: 'final_test',
    kind: 'control',
    label: 'stop → holdout',
    sourceHandle: 'bottom-out',
    targetHandle: 'left-in',
  },
]

export const SIMPLE_LOOP_POSITIONS: Record<
  SimpleLoopNodeId,
  { x: number; y: number }
> = {
  load_data: { x: 0, y: 180 },
  prepare_data: { x: 280, y: 180 },
  hypothesis_generator: { x: 560, y: 180 },
  memory_writer: { x: 1120, y: -80 },
  context_builder: { x: 560, y: -80 },
  train_model: { x: 840, y: 180 },
  evaluate_model: { x: 1120, y: 180 },
  final_test: { x: 1400, y: 180 },
}

export const SIMPLE_LOOP_FLOW = [
  'load_data',
  'prepare_data',
  'hypothesis_generator',
  'train_model',
  'evaluate_model',
  'memory_writer',
  'context_builder',
  'final_test',
] as const

export const GRAPH_NODE_TO_VISUAL: Record<string, SimpleLoopNodeId> = {
  load_data: 'load_data',
  prepare_data: 'prepare_data',
  hypothesis_generator: 'hypothesis_generator',
  memory_writer: 'memory_writer',
  context_builder: 'context_builder',
  train_model: 'train_model',
  evaluate_model: 'evaluate_model',
  final_test: 'final_test',
}
