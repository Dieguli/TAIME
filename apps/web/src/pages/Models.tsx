import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';

import { Dataset, Job, Model, datasetsApi, jobsApi, modelsApi } from '../api/client';
import { Icon } from '../components/Icons';
import {
    SUT_INFO,
    SUT_ORDER,
    SutType,
    formatBytes,
    formatDate,
    formatMetric,
    latestModelBySut,
    metricHighlights,
    statusBadgeClass,
    sutLabel,
    toSutType,
} from '../lib/demo';

export function Models() {
    const [models, setModels] = useState<Model[]>([]);
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [jobs, setJobs] = useState<Job[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const loadModels = async () => {
        setLoading(true);
        setError(null);
        try {
            const [modelData, datasetData, jobData] = await Promise.all([
                modelsApi.list(),
                datasetsApi.list(),
                jobsApi.list(),
            ]);
            setModels(modelData);
            setDatasets(datasetData);
            setJobs(jobData);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load models');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadModels();
    }, []);

    const jobMap = useMemo(() => {
        const map = new Map<number, Job>();
        jobs.forEach((job) => map.set(job.id, job));
        return map;
    }, [jobs]);

    const datasetMap = useMemo(() => {
        const map = new Map<number, Dataset>();
        datasets.forEach((dataset) => map.set(dataset.id, dataset));
        return map;
    }, [datasets]);

    const latestModels = useMemo(() => latestModelBySut(models), [models]);

    return (
        <div className="models page-shell">
            <header className="page-header page-header-row">
                <div>
                    <h1>Models</h1>
                    <p className="text-secondary">
                        Download exported artifacts and trace each one back to its dataset and job.
                    </p>
                </div>
                <button className="btn-secondary" onClick={loadModels} disabled={loading}>
                    <Icon name="refresh" className="btn-icon" />
                    Refresh
                </button>
            </header>

            {error && (
                <div className="notice notice-danger">
                    <Icon name="warning" className="notice-icon" />
                    <span>{error}</span>
                </div>
            )}

            <section className="model-latest-grid">
                {SUT_ORDER.map((sut) => {
                    const model = latestModels.get(sut);
                    return (
                        <LatestModelCard
                            key={sut}
                            sut={sut}
                            model={model}
                            job={model ? jobMap.get(model.job_id) : undefined}
                            dataset={
                                model
                                    ? datasetMap.get(jobMap.get(model.job_id)?.dataset_id ?? -1)
                                    : undefined
                            }
                        />
                    );
                })}
            </section>

            <section className="panel">
                <div className="section-header">
                    <div>
                        <h2>Artifact Catalog</h2>
                        <p className="text-secondary">All exported model versions from completed jobs.</p>
                    </div>
                    <Link className="btn-secondary" to="/jobs">
                        <Icon name="play" className="btn-icon" />
                        New retraining run
                    </Link>
                </div>

                {loading ? (
                    <p className="text-secondary block-copy">Loading models...</p>
                ) : models.length === 0 ? (
                    <p className="text-secondary block-copy">
                        No model artifacts yet. Complete a retraining job to publish the first one.
                    </p>
                ) : (
                    <div className="table-wrapper">
                        <table className="data-table">
                            <thead>
                                <tr>
                                    <th>Artifact</th>
                                    <th>SUT</th>
                                    <th>Trace</th>
                                    <th>Metrics</th>
                                    <th>Size</th>
                                    <th>Created</th>
                                    <th>Actions</th>
                                </tr>
                            </thead>
                            <tbody>
                                {models.map((model) => (
                                    <ModelRow
                                        key={model.id}
                                        model={model}
                                        job={jobMap.get(model.job_id)}
                                        dataset={datasetMap.get(jobMap.get(model.job_id)?.dataset_id ?? -1)}
                                    />
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </section>
        </div>
    );
}

interface LatestModelCardProps {
    sut: SutType;
    model?: Model;
    job?: Job;
    dataset?: Dataset;
}

function LatestModelCard({ sut, model, job, dataset }: LatestModelCardProps) {
    const info = SUT_INFO[sut];
    const highlights = metricHighlights(model?.metrics);

    return (
        <article className={`model-latest-card ${model ? 'is-ready' : 'is-empty'}`}>
            <div className="model-latest-header">
                <div>
                    <span className="sut-domain">{info.domain}</span>
                    <h2>{info.shortLabel}</h2>
                </div>
                <span className="model-chip">{info.model}</span>
            </div>
            {model ? (
                <>
                    <div>
                        <strong>{model.name}</strong>
                        <p className="text-secondary">
                            Job #{model.job_id} - {dataset?.name ?? 'dataset unavailable'}
                        </p>
                    </div>
                    <div className="metric-card-grid metric-card-grid-compact">
                        {highlights.slice(0, 2).map((metric) => (
                            <div key={metric.key} className="metric-card">
                                <span>{metric.key.replace(/_/g, ' ')}</span>
                                <strong>{formatMetric(metric.value)}</strong>
                            </div>
                        ))}
                    </div>
                    <div className="model-card-footer">
                        {job && <span className={statusBadgeClass(job.status)}>{job.status}</span>}
                        <a className="btn-primary" href={modelsApi.downloadUrl(model.id)}>
                            <Icon name="download" className="btn-icon" />
                            Download
                        </a>
                    </div>
                </>
            ) : (
                <>
                    <p className="text-secondary">No exported artifact yet for this SUT.</p>
                    <Link className="btn-secondary" to="/jobs">
                        <Icon name="play" className="btn-icon" />
                        Start retraining
                    </Link>
                </>
            )}
        </article>
    );
}

interface ModelRowProps {
    model: Model;
    job?: Job;
    dataset?: Dataset;
}

function ModelRow({ model, job, dataset }: ModelRowProps) {
    const sut = toSutType(model.sut_type);
    const highlights = metricHighlights(model.metrics);

    return (
        <tr>
            <td>
                <div className="table-title">{model.name}</div>
                <div className="text-secondary table-subtitle">v{model.version}</div>
            </td>
            <td>
                <span className="tag">{sut ? SUT_INFO[sut].shortLabel : sutLabel(model.sut_type)}</span>
            </td>
            <td>
                <div className="table-title">Job #{model.job_id}</div>
                <div className="text-secondary table-subtitle">
                    {dataset?.name ?? 'dataset unavailable'} {job ? `- ${job.status}` : ''}
                </div>
            </td>
            <td>
                <div className="metric-stack">
                    {highlights.length > 0 ? (
                        highlights.slice(0, 3).map((metric) => (
                            <span key={metric.key}>
                                {metric.key.replace(/_/g, ' ')}: {formatMetric(metric.value)}
                            </span>
                        ))
                    ) : (
                        <span>N/A</span>
                    )}
                </div>
            </td>
            <td>{formatBytes(model.file_size)}</td>
            <td>{formatDate(model.created_at)}</td>
            <td>
                <div className="table-actions">
                    <a className="btn-secondary" href={modelsApi.downloadUrl(model.id)}>
                        <Icon name="download" className="btn-icon" />
                        Download
                    </a>
                </div>
            </td>
        </tr>
    );
}
