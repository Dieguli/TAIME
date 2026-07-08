import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

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
    SUT_ORDER,
    SutType,
    formatBytes,
    formatDate,
    formatMetric,
    formatPercent,
    latestDatasetBySut,
    latestJobBySut,
    latestModelBySut,
    metricHighlights,
    statusBadgeClass,
} from '../lib/demo';

import './Dashboard.css';

export function Dashboard() {
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [jobs, setJobs] = useState<Job[]>([]);
    const [models, setModels] = useState<Model[]>([]);
    const [runtime, setRuntime] = useState<RuntimeInfo | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadOverview = async () => {
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
            setError(err instanceof Error ? err.message : 'Failed to load demo state');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadOverview();
    }, []);

    const latestDatasets = useMemo(() => latestDatasetBySut(datasets), [datasets]);
    const latestJobs = useMemo(() => latestJobBySut(jobs), [jobs]);
    const latestModels = useMemo(() => latestModelBySut(models), [models]);
    const activeJobs = jobs.filter((job) => job.status === 'pending' || job.status === 'running');
    const completedJobs = jobs.filter((job) => job.status === 'completed');
    const failedJobs = jobs.filter((job) => job.status === 'failed');

    return (
        <div className="dashboard page-shell">
            <header className="page-header page-header-row">
                <div>
                    <h1>Demo Control Center</h1>
                    <p className="text-secondary">
                        Curated zip datasets, retraining jobs, and exported model artifacts in one
                        operator view.
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

            <section className="flow-strip" aria-label="Demo workflow summary">
                <div className="flow-step">
                    <span className="flow-step-number">1</span>
                    <div>
                        <strong>{datasets.length}</strong>
                        <span>Dataset packages</span>
                    </div>
                </div>
                <div className="flow-step">
                    <span className="flow-step-number">2</span>
                    <div>
                        <strong>{jobs.length}</strong>
                        <span>Retraining jobs</span>
                    </div>
                </div>
                <div className="flow-step">
                    <span className="flow-step-number">3</span>
                    <div>
                        <strong>{completedJobs.length}</strong>
                        <span>Completed runs</span>
                    </div>
                </div>
                <div className="flow-step">
                    <span className="flow-step-number">4</span>
                    <div>
                        <strong>{models.length}</strong>
                        <span>Exported models</span>
                    </div>
                </div>
            </section>

            <section className="dashboard-section">
                <div className="section-header">
                    <div>
                        <h2>SUT Readiness</h2>
                        <p className="text-secondary">
                            Each lane shows whether the curated dataset, a retraining job, and a
                            downloadable artifact are ready.
                        </p>
                    </div>
                    <button className="btn-secondary" onClick={loadOverview} disabled={loading}>
                        <Icon name="refresh" className="btn-icon" />
                        Refresh
                    </button>
                </div>

                <div className="sut-control-grid">
                    {SUT_ORDER.map((sut) => (
                        <SutControlCard
                            key={sut}
                            sut={sut}
                            dataset={latestDatasets.get(sut)}
                            job={latestJobs.get(sut)}
                            model={latestModels.get(sut)}
                        />
                    ))}
                </div>
            </section>

            <section className="dashboard-lower-grid">
                <div className="panel">
                    <div className="section-header">
                        <div>
                            <h2>Recent Jobs</h2>
                            <p className="text-secondary">
                                Active and recent outcomes stay visible during the demo.
                            </p>
                        </div>
                        <span className="tag">{activeJobs.length} active</span>
                    </div>
                    <div className="compact-list">
                        {jobs.slice(0, 5).map((job) => (
                            <div key={job.id} className="compact-row">
                                <div>
                                    <strong>Job #{job.id}</strong>
                                    <span className="text-secondary">{SUT_INFO[job.sut_type as SutType]?.shortLabel ?? job.sut_type}</span>
                                </div>
                                <span className={statusBadgeClass(job.status)}>{job.status}</span>
                                <span className="compact-row-value">{formatPercent(job.progress)}</span>
                            </div>
                        ))}
                        {jobs.length === 0 && (
                            <p className="text-secondary">No retraining jobs have been launched yet.</p>
                        )}
                    </div>
                </div>

                <div className="panel">
                    <div className="section-header">
                        <div>
                            <h2>Model Artifacts</h2>
                            <p className="text-secondary">Latest downloadable outputs and headline metrics.</p>
                        </div>
                        <span className="tag">{failedJobs.length} failed</span>
                    </div>
                    <div className="compact-list">
                        {models.slice(0, 4).map((model) => (
                            <div key={model.id} className="compact-row compact-row-model">
                                <div>
                                    <strong>{model.name}</strong>
                                    <span className="text-secondary">
                                        {formatBytes(model.file_size)} - {formatDate(model.created_at)}
                                    </span>
                                </div>
                                <span className="compact-row-value">
                                    {formatMetric(metricHighlights(model.metrics)[0]?.value)}
                                </span>
                            </div>
                        ))}
                        {models.length === 0 && (
                            <p className="text-secondary">Completed jobs will publish artifacts here.</p>
                        )}
                    </div>
                </div>
            </section>
        </div>
    );
}

interface SutControlCardProps {
    sut: SutType;
    dataset?: Dataset;
    job?: Job;
    model?: Model;
}

function SutControlCard({ sut, dataset, job, model }: SutControlCardProps) {
    const info = SUT_INFO[sut];
    const readyState = model ? 'ready' : job?.status === 'completed' ? 'complete' : dataset ? 'dataset' : 'missing';
    const primaryHref = !dataset ? '/datasets' : job?.status === 'running' || job?.status === 'pending' ? '/jobs' : model ? '/models' : '/jobs';
    const primaryLabel = !dataset
        ? 'Upload package'
        : job?.status === 'running' || job?.status === 'pending'
          ? 'Monitor job'
          : model
            ? 'Open model'
            : 'Start retraining';

    return (
        <article className={`sut-control-card sut-control-card-${readyState}`}>
            <div className="sut-control-header">
                <div>
                    <span className="sut-domain">{info.domain}</span>
                    <h3>{info.shortLabel}</h3>
                </div>
                <span className="model-chip">{info.model}</span>
            </div>

            <p className="text-secondary">{info.description}</p>

            <div className="readiness-lines">
                <ReadinessLine
                    label="Dataset"
                    value={dataset ? `${dataset.name} (${dataset.row_count ?? 'N/A'} rows)` : 'Missing package'}
                    complete={Boolean(dataset)}
                />
                <ReadinessLine
                    label="Latest job"
                    value={job ? `#${job.id} ${job.status}` : 'No run yet'}
                    complete={job?.status === 'completed'}
                    warning={job?.status === 'failed' || job?.status === 'cancelled'}
                />
                <ReadinessLine
                    label="Model"
                    value={model ? `${model.name} v${model.version}` : 'No artifact yet'}
                    complete={Boolean(model)}
                />
            </div>

            {job && (
                <div className="mini-progress">
                    <div className="progress-bar">
                        <div
                            className="progress-bar-fill"
                            style={{ width: `${Math.max(0, Math.min(100, job.progress ?? 0))}%` }}
                        />
                    </div>
                    <span>{formatPercent(job.progress)}</span>
                </div>
            )}

            <div className="sut-card-actions">
                <Link className="btn-primary" to={primaryHref}>
                    {primaryLabel}
                    <Icon name="chevron" className="btn-icon" />
                </Link>
                {model && (
                    <a className="btn-secondary" href={`/api/v1/models/${model.id}/download`}>
                        <Icon name="download" className="btn-icon" />
                        Download
                    </a>
                )}
            </div>
        </article>
    );
}

interface ReadinessLineProps {
    label: string;
    value: string;
    complete: boolean;
    warning?: boolean;
}

function ReadinessLine({ label, value, complete, warning = false }: ReadinessLineProps) {
    const icon = warning ? 'warning' : complete ? 'check' : 'x';
    return (
        <div className={`readiness-line ${complete ? 'is-complete' : ''} ${warning ? 'is-warning' : ''}`}>
            <Icon name={icon} className="readiness-icon" />
            <div>
                <span>{label}</span>
                <strong>{value}</strong>
            </div>
        </div>
    );
}
