import { ChangeEvent, FormEvent, useEffect, useMemo, useState } from 'react';

import {
    Dataset,
    Job,
    Model,
    RuntimeInfo,
    datasetsApi,
    jobsApi,
    modelsApi,
    runtimeApi,
} from '../api/client';
import { Icon } from '../components/Icons';
import { RuntimeBadge } from '../components/RuntimeBadge';
import {
    SUT_INFO,
    SutType,
    formatDate,
    formatKey,
    formatMetric,
    formatPercent,
    metricHighlights,
    statusBadgeClass,
    sutLabel,
    toSutType,
} from '../lib/demo';

type JobFilter = 'all' | 'active' | Job['status'];

const jobFilters: { value: JobFilter; label: string }[] = [
    { value: 'all', label: 'All' },
    { value: 'active', label: 'Active' },
    { value: 'completed', label: 'Completed' },
    { value: 'failed', label: 'Failed' },
    { value: 'cancelled', label: 'Cancelled' },
];

export function Jobs() {
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [jobs, setJobs] = useState<Job[]>([]);
    const [models, setModels] = useState<Model[]>([]);
    const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const [selectedDatasetId, setSelectedDatasetId] = useState<number | ''>('');
    const [sutType, setSutType] = useState<SutType | ''>('');
    const [epochs, setEpochs] = useState(10);
    const [learningRate, setLearningRate] = useState(0.001);
    const [batchSize, setBatchSize] = useState(32);
    const [maxSamples, setMaxSamples] = useState('');
    const [sampleMode, setSampleMode] = useState<'full' | 'limit'>('full');
    const [classificationType, setClassificationType] = useState<'binary' | 'multilabel'>('binary');
    const [device, setDevice] = useState<'auto' | 'cpu' | 'cuda'>('auto');
    const [dropout, setDropout] = useState(0.2);
    const [normType, setNormType] = useState('LayerNorm');
    const [normalizeBefore, setNormalizeBefore] = useState(false);
    const [useReversibleInstanceNorm, setUseReversibleInstanceNorm] = useState(false);
    const [lrSchedulerFactor, setLrSchedulerFactor] = useState(0.5);
    const [lrSchedulerPatience, setLrSchedulerPatience] = useState(5);
    const [weightDecay, setWeightDecay] = useState(0.1);
    const [warmupSteps, setWarmupSteps] = useState(0);
    const [seed, setSeed] = useState(42);
    const [submitting, setSubmitting] = useState(false);
    const [selectedJobId, setSelectedJobId] = useState<number | null>(null);
    const [jobFilter, setJobFilter] = useState<JobFilter>('all');

    const datasetMap = useMemo(() => {
        const map = new Map<number, Dataset>();
        datasets.forEach((dataset) => map.set(dataset.id, dataset));
        return map;
    }, [datasets]);

    const modelByJobId = useMemo(() => {
        const map = new Map<number, Model>();
        models.forEach((model) => map.set(model.job_id, model));
        return map;
    }, [models]);

    const selectedDataset = useMemo(() => {
        if (selectedDatasetId === '') return null;
        return datasetMap.get(Number(selectedDatasetId)) ?? null;
    }, [datasetMap, selectedDatasetId]);

    const selectedJob = useMemo(
        () => jobs.find((job) => job.id === selectedJobId) ?? null,
        [jobs, selectedJobId]
    );

    const filteredJobs = useMemo(() => {
        if (jobFilter === 'all') return jobs;
        if (jobFilter === 'active') {
            return jobs.filter((job) => job.status === 'pending' || job.status === 'running');
        }
        return jobs.filter((job) => job.status === jobFilter);
    }, [jobs, jobFilter]);

    const loadJobs = async () => {
        try {
            const data = await jobsApi.list();
            setJobs(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load jobs');
        }
    };

    const loadModels = async () => {
        try {
            const data = await modelsApi.list();
            setModels(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load models');
        }
    };

    const loadAll = async () => {
        setLoading(true);
        setError(null);
        try {
            const [datasetData, jobData, modelData, runtimeData] = await Promise.all([
                datasetsApi.list(),
                jobsApi.list(),
                modelsApi.list(),
                runtimeApi.get().catch(() => null),
            ]);
            setDatasets(datasetData);
            setJobs(jobData);
            setModels(modelData);
            setRuntime(runtimeData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load retraining state');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadAll();

        const interval = window.setInterval(() => {
            loadJobs();
            loadModels();
        }, 4000);
        return () => window.clearInterval(interval);
    }, []);

    const applyFastPreset = (sut: SutType) => {
        setDevice('auto');
        if (sut === 'healthcare_pc') {
            setEpochs(120);
            setLearningRate(0.1);
            setBatchSize(32);
            setSampleMode('full');
            setMaxSamples('');
        } else if (sut === 'infra_port') {
            setEpochs(10);
            setLearningRate(0.001);
            setBatchSize(64);
            setDropout(0.2);
            setNormType('LayerNorm');
            setNormalizeBefore(false);
            setUseReversibleInstanceNorm(false);
            setLrSchedulerFactor(0.5);
            setLrSchedulerPatience(5);
            setSampleMode('full');
            setMaxSamples('');
        } else if (sut === 'disinfo_fake') {
            setEpochs(3);
            setLearningRate(0.000005);
            setBatchSize(16);
            setWeightDecay(0.1);
            setWarmupSteps(0);
            setSeed(42);
            setSampleMode('limit');
            setMaxSamples('2000');
        } else if (sut === 'disinfo_hate') {
            setEpochs(5);
            setLearningRate(0.00005);
            setBatchSize(16);
            setWeightDecay(0.01);
            setWarmupSteps(500);
            setSeed(42);
            setClassificationType('binary');
            setSampleMode('limit');
            setMaxSamples('2000');
        }
    };

    useEffect(() => {
        if (!selectedDataset) return;
        const sut = toSutType(selectedDataset.sut_type);
        if (!sut) return;
        setSutType(sut);
        applyFastPreset(sut);
    }, [selectedDataset]);

    const handleDatasetChange = (event: ChangeEvent<HTMLSelectElement>) => {
        const value = event.target.value;
        if (!value) {
            setSelectedDatasetId('');
            setSutType('');
            return;
        }
        const datasetId = Number(value);
        setSelectedDatasetId(datasetId);
        const dataset = datasetMap.get(datasetId);
        const sut = dataset ? toSutType(dataset.sut_type) : null;
        if (sut) {
            setSutType(sut);
            applyFastPreset(sut);
        }
    };

    const handleFullDatasetPreset = () => {
        setSampleMode('full');
        setMaxSamples('');
    };

    const handleCreateJob = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!selectedDataset || !sutType) {
            setError('Select a dataset package before starting retraining.');
            return;
        }
        if (device === 'cuda' && !runtime?.cuda_available) {
            setError('GPU was selected, but CUDA is not visible to this runtime.');
            return;
        }

        setSubmitting(true);
        setError(null);
        try {
            const config: Record<string, unknown> = {
                epochs: Math.max(1, Number(epochs)),
                learning_rate: Number(learningRate),
                batch_size: Math.max(1, Number(batchSize)),
                device,
            };

            if (sutType === 'healthcare_pc') {
                config.n_estimators = Math.max(1, Number(epochs));
            }

            if (sutType === 'infra_port') {
                config.dropout = Number(dropout);
                config.norm_type = normType;
                config.normalize_before = normalizeBefore;
                config.use_reversible_instance_norm = useReversibleInstanceNorm;
                config.lr_scheduler_factor = Number(lrSchedulerFactor);
                config.lr_scheduler_patience = Math.trunc(Number(lrSchedulerPatience));
                // Blank = train on every series in the package (the backend default).
                const maxSeries = Math.trunc(Number(maxSamples));
                if (maxSamples && maxSeries > 0) {
                    config.max_series = maxSeries;
                }
            }

            if (sutType === 'disinfo_fake' || sutType === 'disinfo_hate') {
                config.weight_decay = Number(weightDecay);
                config.seed = Math.trunc(Number(seed));
                if (sampleMode === 'limit' && maxSamples) {
                    const parsed = Number(maxSamples);
                    if (!Number.isNaN(parsed) && parsed > 0) {
                        config.max_train_samples = Math.trunc(parsed);
                    }
                }
            }

            if (sutType === 'disinfo_hate') {
                config.warmup_steps = Math.max(0, Math.trunc(Number(warmupSteps)));
                config.classification_type = classificationType;
            }

            const job = await jobsApi.create({
                dataset_id: selectedDataset.id,
                sut_type: sutType,
                config,
            });
            setSelectedJobId(job.id);
            setJobFilter('active');
            await loadJobs();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to create job');
        } finally {
            setSubmitting(false);
        }
    };

    const handleCancel = async (jobId: number) => {
        setError(null);
        try {
            await jobsApi.cancel(jobId);
            await loadJobs();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to cancel job');
        }
    };

    const gpuBlocked = device === 'cuda' && !runtime?.cuda_available;
    const selectedInfo = sutType ? SUT_INFO[sutType] : null;
    const maxSamplesPlaceholder =
        selectedDataset?.row_count !== undefined
            ? `Full dataset (${selectedDataset.row_count} rows)`
            : 'Leave blank for full dataset';

    return (
        <div className="jobs page-shell">
            <header className="page-header page-header-row">
                <div>
                    <h1>Retraining</h1>
                    <p className="text-secondary">
                        Start a demo-safe run, monitor progress, and jump straight to the exported
                        model when it completes.
                    </p>
                </div>
                <RuntimeBadge runtime={runtime} />
            </header>

            {error && (
                <div className="notice notice-danger">
                    <Icon name="warning" className="notice-icon" />
                    <span>{error}</span>
                </div>
            )}

            <section className="retraining-layout">
                <form className="panel job-form-panel" onSubmit={handleCreateJob}>
                    <div className="section-header">
                        <div>
                            <h2>Create Retraining Run</h2>
                            <p className="text-secondary">
                                Choose the dataset first; TAIME fills the SUT defaults.
                            </p>
                        </div>
                    </div>

                    <div className="launcher-step">
                        <span className="launcher-step-index">1</span>
                        <div className="launcher-step-body">
                            <h3>Dataset package</h3>
                            <div className="form-row form-row-tight">
                                <div className="form-group">
                                    <label htmlFor="dataset">Dataset</label>
                                    <select
                                        id="dataset"
                                        value={selectedDatasetId}
                                        onChange={handleDatasetChange}
                                    >
                                        <option value="">Select a dataset...</option>
                                        {datasets.map((dataset) => (
                                            <option key={dataset.id} value={dataset.id}>
                                                {dataset.name} ({sutLabel(dataset.sut_type)})
                                            </option>
                                        ))}
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label htmlFor="sut-type">Model type</label>
                                    <input
                                        id="sut-type"
                                        value={sutType ? SUT_INFO[sutType].label : ''}
                                        readOnly
                                        placeholder="Set by dataset"
                                    />
                                </div>
                            </div>
                            {selectedDataset && selectedInfo && (
                                <div className="selected-dataset-strip">
                                    <strong>{selectedDataset.name}</strong>
                                    <span>{selectedInfo.model}</span>
                                    <span>{selectedDataset.row_count ?? 'N/A'} rows</span>
                                </div>
                            )}
                        </div>
                    </div>

                    <div className="launcher-step">
                        <span className="launcher-step-index">2</span>
                        <div className="launcher-step-body">
                            <h3>Demo preset</h3>
                            <div className="preset-row">
                                <button
                                    type="button"
                                    className="btn-secondary"
                                    onClick={() => sutType && applyFastPreset(sutType)}
                                    disabled={!sutType}
                                >
                                    <Icon name="gauge" className="btn-icon" />
                                    Fast demo preset
                                </button>
                                <button
                                    type="button"
                                    className="btn-secondary"
                                    onClick={handleFullDatasetPreset}
                                    disabled={!sutType}
                                >
                                    Full dataset
                                </button>
                                {selectedInfo && <span className="preset-note">{selectedInfo.fastPreset}</span>}
                            </div>

                            <div className="form-row form-row-tight">
                                <div className="form-group">
                                    <label htmlFor="device">Compute device</label>
                                    <select
                                        id="device"
                                        value={device}
                                        onChange={(event) =>
                                            setDevice(event.target.value as 'auto' | 'cpu' | 'cuda')
                                        }
                                        disabled={!sutType}
                                    >
                                        <option value="auto">Auto</option>
                                        <option value="cuda">GPU (CUDA)</option>
                                        <option value="cpu">CPU</option>
                                    </select>
                                </div>
                                <div className="form-group">
                                    <label htmlFor="epochs">
                                        {sutType === 'healthcare_pc' ? 'Estimators' : 'Epochs'}
                                    </label>
                                    <input
                                        type="number"
                                        id="epochs"
                                        value={epochs}
                                        min={1}
                                        max={200}
                                        onChange={(event) => setEpochs(Number(event.target.value))}
                                        disabled={!sutType}
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="lr">Learning rate</label>
                                    <input
                                        type="number"
                                        id="lr"
                                        value={learningRate}
                                        step={0.000001}
                                        onChange={(event) => setLearningRate(Number(event.target.value))}
                                        disabled={!sutType}
                                    />
                                </div>
                                <div className="form-group">
                                    <label htmlFor="batch">Batch size</label>
                                    <input
                                        type="number"
                                        id="batch"
                                        value={batchSize}
                                        min={1}
                                        onChange={(event) => setBatchSize(Number(event.target.value))}
                                        disabled={!sutType}
                                    />
                                </div>
                            </div>

                            {gpuBlocked && (
                                <div className="notice notice-warning">
                                    <Icon name="warning" className="notice-icon" />
                                    <span>
                                        GPU is selected, but CUDA is not visible. Switch to Auto or CPU, or
                                        relaunch the GPU compose profile.
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>

                    <details className="details-panel">
                        <summary>Advanced parameters</summary>
                        <div className="advanced-grid">
                            {sutType === 'infra_port' && (
                                <>
                                    <div className="form-group">
                                        <label htmlFor="dropout">Dropout</label>
                                        <input
                                            type="number"
                                            id="dropout"
                                            value={dropout}
                                            min={0.2}
                                            max={0.5}
                                            step={0.05}
                                            onChange={(event) => setDropout(Number(event.target.value))}
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="norm-type">Norm type</label>
                                        <select
                                            id="norm-type"
                                            value={normType}
                                            onChange={(event) => setNormType(event.target.value)}
                                        >
                                            <option value="LayerNorm">LayerNorm</option>
                                            <option value="LayerNormNoBias">LayerNormNoBias</option>
                                            <option value="TimeBatchNorm2d">TimeBatchNorm2d</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="lr-factor">LR scheduler factor</label>
                                        <input
                                            type="number"
                                            id="lr-factor"
                                            value={lrSchedulerFactor}
                                            min={0.1}
                                            max={0.9}
                                            step={0.1}
                                            onChange={(event) =>
                                                setLrSchedulerFactor(Number(event.target.value))
                                            }
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="lr-patience">LR scheduler patience</label>
                                        <input
                                            type="number"
                                            id="lr-patience"
                                            value={lrSchedulerPatience}
                                            min={1}
                                            max={10}
                                            step={1}
                                            onChange={(event) =>
                                                setLrSchedulerPatience(Number(event.target.value))
                                            }
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="max-series">Max series (port calls)</label>
                                        <input
                                            type="number"
                                            id="max-series"
                                            value={maxSamples}
                                            min={1}
                                            step={1}
                                            onChange={(event) => setMaxSamples(event.target.value)}
                                            placeholder="All series"
                                        />
                                    </div>
                                    <label className="toggle-row" htmlFor="rev-in">
                                        <input
                                            type="checkbox"
                                            id="rev-in"
                                            checked={useReversibleInstanceNorm}
                                            onChange={(event) =>
                                                setUseReversibleInstanceNorm(event.target.checked)
                                            }
                                        />
                                        Reversible instance norm
                                    </label>
                                    <label className="toggle-row" htmlFor="norm-before">
                                        <input
                                            type="checkbox"
                                            id="norm-before"
                                            checked={normalizeBefore}
                                            onChange={(event) => setNormalizeBefore(event.target.checked)}
                                        />
                                        Normalize before
                                    </label>
                                </>
                            )}

                            {(sutType === 'disinfo_fake' || sutType === 'disinfo_hate') && (
                                <>
                                    <div className="form-group">
                                        <label htmlFor="weight-decay">Weight decay</label>
                                        <input
                                            type="number"
                                            id="weight-decay"
                                            value={weightDecay}
                                            min={0}
                                            step={0.01}
                                            onChange={(event) => setWeightDecay(Number(event.target.value))}
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="seed">Seed</label>
                                        <input
                                            type="number"
                                            id="seed"
                                            value={seed}
                                            min={0}
                                            step={1}
                                            onChange={(event) => setSeed(Number(event.target.value))}
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="sample-mode">Sample mode</label>
                                        <select
                                            id="sample-mode"
                                            value={sampleMode}
                                            onChange={(event) =>
                                                setSampleMode(event.target.value as 'full' | 'limit')
                                            }
                                        >
                                            <option value="limit">Limit samples</option>
                                            <option value="full">Full dataset</option>
                                        </select>
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="max-samples">Max samples</label>
                                        <input
                                            type="number"
                                            id="max-samples"
                                            value={maxSamples}
                                            min={1}
                                            onChange={(event) => setMaxSamples(event.target.value)}
                                            placeholder={maxSamplesPlaceholder}
                                            disabled={sampleMode === 'full'}
                                        />
                                    </div>
                                </>
                            )}

                            {sutType === 'disinfo_hate' && (
                                <>
                                    <div className="form-group">
                                        <label htmlFor="warmup">Warmup steps</label>
                                        <input
                                            type="number"
                                            id="warmup"
                                            value={warmupSteps}
                                            min={0}
                                            step={50}
                                            onChange={(event) => setWarmupSteps(Number(event.target.value))}
                                        />
                                    </div>
                                    <div className="form-group">
                                        <label htmlFor="classification">Classification mode</label>
                                        <select
                                            id="classification"
                                            value={classificationType}
                                            onChange={(event) =>
                                                setClassificationType(
                                                    event.target.value as 'binary' | 'multilabel'
                                                )
                                            }
                                        >
                                            <option value="binary">Binary</option>
                                            <option value="multilabel">Multilabel</option>
                                        </select>
                                    </div>
                                </>
                            )}

                            {!sutType && (
                                <p className="text-secondary">Select a dataset to unlock parameters.</p>
                            )}
                        </div>
                    </details>

                    <button
                        type="submit"
                        className="btn-primary launcher-submit"
                        disabled={submitting || !selectedDataset || gpuBlocked}
                    >
                        <Icon name="play" className="btn-icon" />
                        {submitting ? 'Starting...' : 'Start retraining'}
                    </button>
                </form>

                <aside className="panel runtime-panel">
                    <h2>Runtime Guardrail</h2>
                    <RuntimeBadge runtime={runtime} />
                    <div className="runtime-detail-list">
                        <span>Image: {runtime?.image ?? 'unknown'}</span>
                        <span>CUDA: {runtime?.cuda_available ? 'available' : 'not available'}</span>
                        <span>PyTorch: {runtime?.torch_version ?? 'not detected'}</span>
                    </div>
                    <p className="text-secondary">
                        Use Auto for demos unless you specifically need to prove the GPU path.
                    </p>
                </aside>
            </section>

            <section className="panel">
                <div className="section-header">
                    <div>
                        <h2>Job History</h2>
                        <p className="text-secondary">Filter active runs, inspect failures, or download outputs.</p>
                    </div>
                    <button className="btn-secondary" onClick={loadAll} disabled={loading}>
                        <Icon name="refresh" className="btn-icon" />
                        Refresh
                    </button>
                </div>

                <div className="segmented-control" role="tablist" aria-label="Job status filter">
                    {jobFilters.map((filter) => (
                        <button
                            key={filter.value}
                            type="button"
                            className={jobFilter === filter.value ? 'is-selected' : ''}
                            onClick={() => setJobFilter(filter.value)}
                        >
                            {filter.label}
                        </button>
                    ))}
                </div>

                {loading ? (
                    <p className="text-secondary block-copy">Loading jobs...</p>
                ) : filteredJobs.length === 0 ? (
                    <p className="text-secondary block-copy">No jobs match this filter.</p>
                ) : (
                    <div className="table-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Run</th>
                                    <th>Dataset</th>
                                    <th>Status</th>
                                    <th>Progress</th>
                                    <th>Epoch</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {filteredJobs.map((job) => (
                                    <JobRow
                                        key={job.id}
                                        job={job}
                                        dataset={datasetMap.get(job.dataset_id)}
                                        model={modelByJobId.get(job.id)}
                                        onSelect={setSelectedJobId}
                                        onCancel={handleCancel}
                                    />
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </section>

            {selectedJob && (
                <JobReport
                    job={selectedJob}
                    dataset={datasetMap.get(selectedJob.dataset_id)}
                    model={modelByJobId.get(selectedJob.id)}
                    onClose={() => setSelectedJobId(null)}
                />
            )}
        </div>
    );
}

interface JobRowProps {
    job: Job;
    dataset?: Dataset;
    model?: Model;
    onSelect: (jobId: number) => void;
    onCancel: (jobId: number) => void;
}

function JobRow({ job, dataset, model, onSelect, onCancel }: JobRowProps) {
    const canCancel = job.status === 'pending' || job.status === 'running';
    return (
        <tr>
            <td>
                <div className="table-title">#{job.id}</div>
                <div className="text-secondary table-subtitle">{sutLabel(job.sut_type)}</div>
            </td>
            <td>
                <div className="table-title">{dataset?.name ?? 'Dataset unavailable'}</div>
                <div className="text-secondary table-subtitle">{dataset?.row_count ?? 'N/A'} rows</div>
            </td>
            <td>
                <span className={statusBadgeClass(job.status)}>{job.status}</span>
            </td>
            <td>
                <div className="progress-bar">
                    <div
                        className="progress-bar-fill"
                        style={{ width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%` }}
                    />
                </div>
                <div className="text-secondary table-subtitle">{formatPercent(job.progress)}</div>
            </td>
            <td>
                {job.current_epoch ?? 0}/{job.total_epochs ?? 0}
            </td>
            <td>{formatDate(job.created_at)}</td>
            <td>
                <div className="table-actions">
                    <button className="btn-secondary" onClick={() => onSelect(job.id)}>
                        <Icon name="file" className="btn-icon" />
                        Report
                    </button>
                    {model && (
                        <a className="btn-secondary" href={`/api/v1/models/${model.id}/download`}>
                            <Icon name="download" className="btn-icon" />
                            Download
                        </a>
                    )}
                    {canCancel && (
                        <button className="btn-danger" onClick={() => onCancel(job.id)}>
                            Cancel
                        </button>
                    )}
                </div>
            </td>
        </tr>
    );
}

interface JobReportProps {
    job: Job;
    dataset?: Dataset;
    model?: Model;
    onClose: () => void;
}

function JobReport({ job, dataset, model, onClose }: JobReportProps) {
    const highlights = metricHighlights(job.metrics);

    return (
        <section className="panel report-panel">
            <div className="section-header">
                <div>
                    <h2>Job Report</h2>
                    <p className="text-secondary">
                        Job #{job.id} - {sutLabel(job.sut_type)}
                    </p>
                </div>
                <div className="table-actions">
                    {model && (
                        <a className="btn-primary" href={`/api/v1/models/${model.id}/download`}>
                            <Icon name="download" className="btn-icon" />
                            Download model
                        </a>
                    )}
                    <button className="btn-secondary" onClick={onClose}>
                        <Icon name="x" className="btn-icon" />
                        Close
                    </button>
                </div>
            </div>

            <div className="report-summary-grid">
                <div className="report-card">
                    <span className="summary-label">Status</span>
                    <span className={statusBadgeClass(job.status)}>{job.status}</span>
                    <p className="text-secondary">Progress: {formatPercent(job.progress)}</p>
                </div>
                <div className="report-card">
                    <span className="summary-label">Dataset</span>
                    <strong>{dataset?.name ?? 'Dataset unavailable'}</strong>
                    <p className="text-secondary">{dataset?.row_count ?? 'N/A'} rows</p>
                </div>
                <div className="report-card">
                    <span className="summary-label">Timing</span>
                    <p className="text-secondary">Started: {formatDate(job.started_at)}</p>
                    <p className="text-secondary">Completed: {formatDate(job.completed_at)}</p>
                </div>
                <div className="report-card">
                    <span className="summary-label">Artifact</span>
                    <strong>{model ? model.name : 'Not exported yet'}</strong>
                    <p className="text-secondary">{model ? `v${model.version}` : 'Completes after a successful run.'}</p>
                </div>
            </div>

            {job.error_message && (
                <div className="notice notice-danger">
                    <Icon name="warning" className="notice-icon" />
                    <span>{job.error_message}</span>
                </div>
            )}

            <div className="metric-card-grid">
                {highlights.length > 0 ? (
                    highlights.map((metric) => (
                        <div key={metric.key} className="metric-card">
                            <span>{formatKey(metric.key)}</span>
                            <strong>{formatMetric(metric.value)}</strong>
                        </div>
                    ))
                ) : (
                    <p className="text-secondary">Metrics will appear when the job reports results.</p>
                )}
            </div>

            <details className="details-panel">
                <summary>Technical config and metrics</summary>
                <div className="json-grid">
                    <div>
                        <h3>Configuration</h3>
                        <pre className="stat-pre">{JSON.stringify(job.config ?? {}, null, 2)}</pre>
                    </div>
                    <div>
                        <h3>Metrics</h3>
                        <pre className="stat-pre">{JSON.stringify(job.metrics ?? {}, null, 2)}</pre>
                    </div>
                </div>
            </details>
        </section>
    );
}
