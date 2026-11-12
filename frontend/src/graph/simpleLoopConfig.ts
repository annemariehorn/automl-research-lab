
/** NYC Taxi LangGraph workflow configuration. */

export type SimpleLoopNodeId =
  | 'load_data'
  | 'prepare_data'
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
    role: 'Read taxi trip and zone datasets',
    kind: 'file',
    service: 'data ingestion',
  },
  {
    id: 'prepare_data',
    title: 'Prepare Data',
    role: 'Clean, transform and engineer features',
    kind: 'code',
    service: 'data preprocessing',
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
    id: 'e-prepare-train',
    source: 'prepare_data',
    target: 'train_model',
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
    id: 'e-evaluate-retry',
    source: 'evaluate_model',
    target: 'train_model',
    kind: 'control',
    label: 'retry',
    sourceHandle: 'top-out',
    targetHandle: 'top-in',
  },
  {
    id: 'e-evaluate-final',
    source: 'evaluate_model',
    target: 'final_test',
    kind: 'control',
    label: 'accepted',
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
  train_model: { x: 560, y: 180 },
  evaluate_model: { x: 840, y: 180 },
  final_test: { x: 1120, y: 180 },
}

export const SIMPLE_LOOP_FLOW = [
  'load_data',
  'prepare_data',
  'train_model',
  'evaluate_model',
  'final_test',
] as const

export const GRAPH_NODE_TO_VISUAL: Record<string, SimpleLoopNodeId> = {
  load_data: 'load_data',
  prepare_data: 'prepare_data',
  train_model: 'train_model',
  evaluate_model: 'evaluate_model',
  final_test: 'final_test',
}
