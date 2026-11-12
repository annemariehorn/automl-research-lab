
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link } from 'react-router-dom'

import { getLabLoopStatus } from '../api/gateway'
import { LiveSimpleLoopGraph, visualEdgeKeysFromLangGraph, visualIdsFromLangGraphNodes } from '../components/LiveSimpleLoopGraph'
import type { SimpleLoopRuntime, SimpleNodeStatus } from '../components/LiveSimpleLoopGraph'
import { ScoreChart } from '../components/ScoreChart'
import { ExperimentTimeline } from '../components/ExperimentTimeline'
import { BestVsPrevious } from '../components/BestVsPrevious'
import { BudgetPanel } from '../components/BudgetPanel'
import { SessionMeta } from '../components/SessionMeta'
import { initialSessionState } from '../state/sessionReducer'
import type { ExperimentEntry, SessionState } from '../types/events'

const TAXI_DATASET = '/packages/artifacts/nyc_taxi/taxi_ml_ready.parquet'
const DATASET_NAME = 'taxi_ml_ready.parquet'
const TARGET_COLUMN = 'fare_amount'

const METRIC = 'f1'
const STRATEGY = 'model_search' as const
const TRAIN_TIMEOUT = 120
const SELF_HEAL_BUDGET = 3

type TaxiOutput = {
  dataset_path?: string
  row_count?: number
  model_name?: string
  accuracy?: number
  f1_score?: number
  cycle?: number
  best_model?: string
  best_f1_score?: number
  decision?: string
  stop_reason?: string
  test_accuracy?: number
  test_f1_score?: number
}

type TaxiResult = TaxiOutput & {
  max_cycles?: number
  target_score?: number
  history?: Array<{
    cycle: number
    model: string
    accuracy: number
    f1_score: number
  }>
}

type StreamEvent = {
  type: string
  node?: string
  output?: TaxiOutput
  result?: TaxiResult
  message?: string
}

type GraphStatus = {
  ready: boolean
  available?: boolean
  nodes: string[]
  edges: Array<
    { source: string; target: string } | [string, string]
  >
  error?: string | null
  missing_nodes?: string[]
}

function setStatus(
  prev: SimpleLoopRuntime,
  id: string,
  status: SimpleNodeStatus,
  note?: string,
): SimpleLoopRuntime {
  return {
    ...prev,
    [id]: {
      status,
      note: note ?? prev[id]?.note,
    },
  }
}

export function LabLoopPage() {
  const [graphStatus, setGraphStatus] = useState<GraphStatus | null>(null)
  const [statusLoading, setStatusLoading] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [targetScore, setTargetScore] = useState(0.99)
  const [maxCycles, setMaxCycles] = useState(3)

  const [runtime, setRuntime] = useState<SimpleLoopRuntime>({})
  const [activeNodeId, setActiveNodeId] = useState<string | null>(null)

  const [experiments, setExperiments] = useState<ExperimentEntry[]>([])
  const [liveBest, setLiveBest] = useState<number | null>(null)
  const [liveCycle, setLiveCycle] = useState(0)
  const [stopReason, setStopReason] = useState<string | null>(null)
  const [holdoutScore, setHoldoutScore] = useState<number | null>(null)
  const [result, setResult] = useState<TaxiResult | null>(null)

  const sourceRef = useRef<EventSource | null>(null)
  const runningRef = useRef(false)

  const refreshGraphStatus = useCallback(async () => {
    setStatusLoading(true)

    try {
      const status = await getLabLoopStatus()
      setGraphStatus(status as GraphStatus)
    } catch (err) {
      setGraphStatus({
        ready: false,
        nodes: [],
        edges: [],
        error:
          err instanceof Error
            ? err.message
            : 'Failed to load graph status',
      })
    } finally {
      setStatusLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshGraphStatus()
  }, [refreshGraphStatus])

  useEffect(() => {
    return () => {
      sourceRef.current?.close()
    }
  }, [])

  const resetRuntime = useCallback(() => {
    sourceRef.current?.close()
    sourceRef.current = null
    runningRef.current = false

    setBusy(false)
    setError(null)
    setResult(null)
    setRuntime({})
    setActiveNodeId(null)
    setExperiments([])
    setLiveBest(null)
    setLiveCycle(0)
    setStopReason(null)
    setHoldoutScore(null)
  }, [])

  const onRun = useCallback(() => {
    if (!graphStatus?.ready) {
      setError('NYC Taxi graph is not ready')
      return
    }

    resetRuntime()

    setBusy(true)
    runningRef.current = true

    const params = new URLSearchParams({
      dataset_path: TAXI_DATASET,
      target_column: TARGET_COLUMN,
      target_score: String(targetScore),
      max_cycles: String(maxCycles),
    })

    const source = new EventSource(
      `/api/lab-loop/stream?${params.toString()}`,
    )

    sourceRef.current = source

    const finish = () => {
      runningRef.current = false
      setBusy(false)
      setActiveNodeId(null)
      source.close()

      if (sourceRef.current === source) {
        sourceRef.current = null
      }
    }

    const handle = (msg: MessageEvent<string>) => {
      let ev: StreamEvent

      try {
        ev = JSON.parse(msg.data) as StreamEvent
      } catch {
        return
      }

      if (ev.type === 'run_started') {
        setError(null)
        return
      }

      if (ev.type === 'node_started' && ev.node) {
        const node = ev.node

        setActiveNodeId(node)

        setRuntime((prev) =>
          setStatus(prev, node, 'running'),
        )

        return
      }

      if (ev.type === 'node_completed' && ev.node) {
        const node = ev.node
        const output = ev.output ?? {}

        let note: string | undefined

        if (node === 'load_data') {
          if (typeof output.row_count === 'number') {
            note = `${output.row_count.toLocaleString()} rows loaded`
          }
        }

        if (node === 'prepare_data') {
          note = 'Training and test data prepared'
        }

        if (node === 'train_model') {
          if (
            typeof output.f1_score === 'number' &&
            typeof output.cycle === 'number'
          ) {
            const cycle = output.cycle
            const f1 = output.f1_score
            const accuracy = output.accuracy ?? 0
            const modelName = output.model_name ?? 'Model'

            note = `${modelName}: F1 ${f1.toFixed(4)}`

            const experiment: ExperimentEntry = {
              experimentId: `exp-${cycle}`,
              modelFamily: modelName,
              status: 'completed',
              trainMetrics: {},
              validationMetrics: {
                f1,
                accuracy,
              },
              note: `${modelName} — validation F1 ${f1.toFixed(4)}`,
              agentOutputs: {},
            }

            setExperiments((prev) => {
              const without = prev.filter(
                (item) =>
                  item.experimentId !== experiment.experimentId,
              )

              return [...without, experiment]
            })

            setLiveCycle(cycle)

            setLiveBest((prev) =>
              Math.max(prev ?? 0, f1),
            )
          }
        }

        if (node === 'evaluate_model') {
          if (typeof output.best_f1_score === 'number') {
            setLiveBest(output.best_f1_score)
          }

          note = output.decision
            ? `Decision: ${output.decision}`
            : 'Model evaluated'

          if (output.stop_reason) {
            setStopReason(output.stop_reason)
          }
        }

        if (node === 'final_test') {
          if (typeof output.test_f1_score === 'number') {
            setHoldoutScore(output.test_f1_score)
          }

          note =
            typeof output.test_accuracy === 'number'
              ? `Test accuracy: ${output.test_accuracy.toFixed(4)}`
              : 'Final test completed'
        }

        setRuntime((prev) =>
          setStatus(prev, node, 'completed', note),
        )

        return
      }

      if (ev.type === 'run_completed') {
        if (ev.result) {
          const finalResult = ev.result

          setResult(finalResult)

          if (typeof finalResult.cycle === 'number') {
            setLiveCycle(finalResult.cycle)
          }

          if (
            typeof finalResult.best_f1_score === 'number'
          ) {
            setLiveBest(finalResult.best_f1_score)
          }

          if (
            typeof finalResult.test_f1_score === 'number'
          ) {
            setHoldoutScore(finalResult.test_f1_score)
          }

          if (finalResult.stop_reason) {
            setStopReason(finalResult.stop_reason)
          }

          if (finalResult.history?.length) {
            setExperiments(
              finalResult.history.map((entry) => ({
                experimentId: `exp-${entry.cycle}`,
                modelFamily: entry.model,
                status: 'completed',
                trainMetrics: {},
                validationMetrics: {
                  f1: entry.f1_score,
                  accuracy: entry.accuracy,
                },
                note: `${entry.model} — validation F1 ${entry.f1_score.toFixed(4)}`,
                agentOutputs: {},
              })),
            )
          }
        }

        finish()
        return
      }

      if (ev.type === 'error') {
        setError(ev.message || 'NYC Taxi workflow failed')
        finish()
      }
    }

    const eventTypes = [
      'run_started',
      'node_started',
      'node_completed',
      'run_completed',
      'error',
    ]

    eventTypes.forEach((type) => {
      source.addEventListener(
        type,
        handle as EventListener,
      )
    })

    source.onerror = () => {
      if (runningRef.current) {
        setError('Connection to the NYC Taxi workflow was lost')
      }

      finish()
    }
  }, [
    graphStatus?.ready,
    targetScore,
    maxCycles,
    resetRuntime,
  ])

  const ready = Boolean(graphStatus?.ready)
  const canRun = ready && !busy

  const visibleNodeIds = useMemo(
    () =>
      Array.from(
        visualIdsFromLangGraphNodes(
          graphStatus?.nodes ?? [],
        ),
      ),
    [graphStatus?.nodes],
  )

  const visibleEdgeKeys = useMemo(() => {
    const edges = graphStatus?.edges ?? []

    return visualEdgeKeysFromLangGraph(
      edges.map((edge) =>
        Array.isArray(edge)
          ? { source: edge[0], target: edge[1] }
          : edge,
      ),
    )
  }, [graphStatus?.edges])

  const panelState: SessionState = useMemo(
    () =>
      initialSessionState({
        sessionId: 'nyc-taxi-lab-loop',
        status: busy
          ? 'running'
          : result
            ? 'completed'
            : error
              ? 'error'
              : 'idle',
        mode: 'live',
        dataset: DATASET_NAME,
        targetColumn: TARGET_COLUMN,
        metric: METRIC,
        targetScore,
        experimentBudget: maxCycles,
        experimentsUsed: experiments.length,
        selfHealBudget: SELF_HEAL_BUDGET,
        researchStrategy: STRATEGY,
        perTrialTimeoutSeconds: TRAIN_TIMEOUT,
        bestValScore: liveBest,
        holdoutScore,
        stopReason,
        experiments,
        traceEvents: [],
        error,
      }),
    [
      busy,
      result,
      error,
      targetScore,
      maxCycles,
      experiments,
      liveBest,
      holdoutScore,
      stopReason,
    ],
  )

  return (
    <div className="lab-grid flex min-h-screen flex-col">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-line px-6 py-4">
        <div className="flex min-w-0 items-start gap-3">
          <Link
            to="/guide"
            className="mt-1 inline-flex shrink-0 items-center gap-1.5 rounded border border-line bg-panel px-3 py-2 text-sm text-paper hover:border-teal"
          >
            ← Back
          </Link>

          <div>
            <p className="font-mono text-[10px] uppercase tracking-wider text-teal">
              Lab
            </p>

            <h1 className="font-display text-3xl font-semibold text-paper">
              NYC Taxi ML Loop
            </h1>

            <p className="mt-1 text-sm text-mist">
              Live LangGraph workflow
              {liveCycle > 0
                ? ` · Cycle ${liveCycle}`
                : ''}
            </p>
          </div>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <button
            type="button"
            onClick={() => void refreshGraphStatus()}
            disabled={statusLoading || busy}
            className="rounded border border-line bg-panel px-3 py-2 text-sm text-paper hover:border-teal disabled:opacity-50"
          >
            {statusLoading
              ? 'Checking…'
              : 'Refresh graph'}
          </button>

          <label className="flex items-center gap-2 text-sm text-mist">
            Target F1

            <input
              type="number"
              step="0.01"
              min={0.5}
              max={0.99}
              value={targetScore}
              onChange={(e) =>
                setTargetScore(Number(e.target.value))
              }
              disabled={busy}
              className="w-20 border border-line bg-ink-2 px-2 py-1.5 font-mono text-paper"
            />
          </label>

          <label className="flex items-center gap-2 text-sm text-mist">
            Budget

            <input
              type="number"
              min={1}
              max={8}
              value={maxCycles}
              onChange={(e) =>
                setMaxCycles(Number(e.target.value))
              }
              disabled={busy}
              className="w-16 border border-line bg-ink-2 px-2 py-1.5 font-mono text-paper"
            />
          </label>

          <span className="rounded border border-line px-2 py-1 font-mono text-xs text-mist">
            {!ready
              ? 'Not ready'
              : busy
                ? 'Running'
                : result
                  ? 'Completed'
                  : 'Ready'}
          </span>

          <button
            type="button"
            disabled={!canRun}
            onClick={onRun}
            className="rounded border border-teal bg-teal px-4 py-2 text-sm font-medium text-white disabled:opacity-50"
          >
            {busy ? 'Running…' : 'Run loop'}
          </button>

          {(result || error) && (
            <button
              type="button"
              disabled={busy}
              onClick={resetRuntime}
              className="rounded border border-line bg-panel px-3 py-2 text-sm text-paper hover:border-teal"
            >
              Reset
            </button>
          )}
        </div>
      </header>

      <div className="flex flex-wrap items-center gap-6 border-b border-line px-6 py-3 text-sm text-mist">
        <div>
          <span className="font-mono text-xs uppercase">
            Dataset:
          </span>{' '}
          <span className="text-paper">
            {DATASET_NAME}
          </span>
        </div>

        <div>
          <span className="font-mono text-xs uppercase">
            Target:
          </span>{' '}
          <span className="text-paper">
            {TARGET_COLUMN}
          </span>
        </div>

        <div>
          <span className="font-mono text-xs uppercase">
            Models:
          </span>{' '}
          <span className="text-paper">
            Logistic Regression, Decision Tree, Random Forest
          </span>
        </div>
      </div>

      <SessionMeta state={panelState} />

      {error && (
        <p className="border-b border-danger/40 bg-danger/10 px-6 py-2 text-sm text-danger">
          {error}
        </p>
      )}

      {!ready && graphStatus?.error && (
        <p className="border-b border-danger/40 px-6 py-2 text-sm text-danger">
          {graphStatus.error}
        </p>
      )}

      <div className="mx-auto grid w-full max-w-[1400px] flex-1 gap-6 px-4 py-6 lg:grid-cols-[1.1fr_0.9fr]">
        <div className="flex min-h-0 flex-col gap-6">
          <LiveSimpleLoopGraph
            runtime={runtime}
            activeNodeId={activeNodeId}
            datasetLabel={DATASET_NAME}
            targetColumn={TARGET_COLUMN}
            visibleNodeIds={visibleNodeIds}
            visibleEdgeKeys={visibleEdgeKeys}
          />

          <ExperimentTimeline
            experiments={experiments}
            traceEvents={[]}
          />
        </div>

        <div className="flex min-h-0 flex-col gap-6">
          <ScoreChart
            experiments={experiments}
            targetScore={targetScore}
          />

          <BestVsPrevious
            experiments={experiments}
          />

          <BudgetPanel state={panelState} />

          {result && (
            <div className="rounded border border-line bg-panel p-5">
              <h2 className="mb-4 text-lg font-semibold text-paper">
                Final NYC Taxi Results
              </h2>

              <div className="space-y-3 text-sm">
                <p>
                  <span className="text-mist">
                    Rows analyzed:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.row_count?.toLocaleString() ?? '—'}
                  </strong>
                </p>

                <p>
                  <span className="text-mist">
                    Training cycles:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.cycle ?? '—'}
                  </strong>
                </p>

                <p>
                  <span className="text-mist">
                    Best model:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.best_model ?? '—'}
                  </strong>
                </p>

                <p>
                  <span className="text-mist">
                    Best validation F1:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.best_f1_score?.toFixed(4) ?? '—'}
                  </strong>
                </p>

                <p>
                  <span className="text-mist">
                    Test accuracy:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.test_accuracy?.toFixed(4) ?? '—'}
                  </strong>
                </p>

                <p>
                  <span className="text-mist">
                    Test F1:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.test_f1_score?.toFixed(4) ?? '—'}
                  </strong>
                </p>

                <p>
                  <span className="text-mist">
                    Stop reason:
                  </span>{' '}
                  <strong className="text-paper">
                    {result.stop_reason ?? '—'}
                  </strong>
                </p>
              </div>
            </div>
          )}
        </div>
      </div>

      <footer className="border-t border-line px-6 py-3 text-center text-xs text-mist">
        <Link to="/guide" className="text-teal hover:underline">
          Guide
        </Link>

        {' · '}

        <Link
          to="/solution-loop"
          className="text-teal hover:underline"
        >
          Solution loop
        </Link>

        {' · '}

        NYC Taxi live agent graph driven by LangGraph SSE
      </footer>
    </div>
  )
}
