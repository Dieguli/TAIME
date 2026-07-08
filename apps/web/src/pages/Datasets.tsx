import { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';

import { Dataset, DatasetPreview, datasetsApi } from '../api/client';
import { Icon } from '../components/Icons';
import {
    SUT_INFO,
    SUT_ORDER,
    SutType,
    formatBytes,
    formatDate,
    formatKey,
    latestDatasetBySut,
    toSutType,
} from '../lib/demo';

const renderStatValue = (value: unknown) => {
    if (value === null || value === undefined) return <span>N/A</span>;
    if (typeof value === 'number') {
        const rounded = Number.isInteger(value) ? value.toString() : value.toFixed(4);
        return <span>{rounded}</span>;
    }
    if (typeof value === 'string') return <span>{value}</span>;
    if (Array.isArray(value)) return <span>{value.join(', ')}</span>;
    return <pre className="stat-pre">{JSON.stringify(value, null, 2)}</pre>;
};

const renderColumns = (columns: string[]) => {
    if (!columns.length) return 'N/A';
    const limit = 12;
    if (columns.length <= limit) return columns.join(', ');
    const visible = columns.slice(0, limit).join(', ');
    return `${visible} (+${columns.length - limit} more)`;
};

export function Datasets() {
    const [datasets, setDatasets] = useState<Dataset[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [preview, setPreview] = useState<DatasetPreview | null>(null);
    const [previewLoadingId, setPreviewLoadingId] = useState<number | null>(null);

    const [name, setName] = useState('');
    const [description, setDescription] = useState('');
    const [sutType, setSutType] = useState<SutType>('healthcare_pc');
    const [file, setFile] = useState<File | null>(null);
    const [uploading, setUploading] = useState(false);
    const fileInputRef = useRef<HTMLInputElement | null>(null);

    const selectedSut = SUT_INFO[sutType];
    const latestDatasets = useMemo(() => latestDatasetBySut(datasets), [datasets]);

    const loadDatasets = async () => {
        setLoading(true);
        setError(null);
        try {
            const data = await datasetsApi.list();
            setDatasets(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to load datasets');
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        loadDatasets();
    }, []);

    const handleUpload = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!file) {
            setError('Select a dataset zip to upload.');
            return;
        }
        if (!file.name.toLowerCase().endsWith('.zip')) {
            setError('Only .zip uploads are supported.');
            return;
        }
        if (!name.trim()) {
            setError('Provide a dataset name.');
            return;
        }
        setUploading(true);
        setError(null);
        try {
            await datasetsApi.upload(file, name.trim(), sutType, description.trim() || undefined);
            setName('');
            setDescription('');
            setFile(null);
            setPreview(null);
            if (fileInputRef.current) {
                fileInputRef.current.value = '';
            }
            await loadDatasets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Upload failed');
        } finally {
            setUploading(false);
        }
    };

    const handlePreview = async (datasetId: number) => {
        setPreviewLoadingId(datasetId);
        setError(null);
        try {
            const data = await datasetsApi.preview(datasetId);
            setPreview(data);
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Preview failed');
        } finally {
            setPreviewLoadingId(null);
        }
    };

    const handleDelete = async (datasetId: number) => {
        setError(null);
        try {
            await datasetsApi.delete(datasetId);
            if (preview?.id === datasetId) {
                setPreview(null);
            }
            await loadDatasets();
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Delete failed');
        }
    };

    return (
        <div className="datasets page-shell">
            <header className="page-header page-header-row">
                <div>
                    <h1>Datasets</h1>
                    <p className="text-secondary">
                        Upload curated zip packages and confirm each demo lane is ready to retrain.
                    </p>
                </div>
                <button className="btn-secondary" onClick={loadDatasets} disabled={loading}>
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

            <section className="readiness-grid">
                {SUT_ORDER.map((sut) => {
                    const dataset = latestDatasets.get(sut);
                    const info = SUT_INFO[sut];
                    return (
                        <article key={sut} className="readiness-card">
                            <div>
                                <span className="sut-domain">{info.domain}</span>
                                <h2>{info.shortLabel}</h2>
                            </div>
                            <span className={dataset ? 'badge badge-completed' : 'badge badge-pending'}>
                                {dataset ? 'ready' : 'needed'}
                            </span>
                            <p className="text-secondary">
                                {dataset ? `${dataset.name} - ${dataset.row_count ?? 'N/A'} rows` : info.fastPreset}
                            </p>
                        </article>
                    );
                })}
            </section>

            <section className="split-panel">
                <div className="panel">
                    <div className="section-header">
                        <div>
                            <h2>Upload Package</h2>
                            <p className="text-secondary">Only zip packages matching the selected SUT are accepted.</p>
                        </div>
                    </div>

                    <form className="upload-form" onSubmit={handleUpload}>
                        <div className="form-row">
                            <div className="form-group">
                                <label htmlFor="dataset-name">Dataset name</label>
                                <input
                                    id="dataset-name"
                                    value={name}
                                    onChange={(event) => setName(event.target.value)}
                                    placeholder="e.g., Fake News Demo"
                                />
                            </div>
                            <div className="form-group">
                                <label htmlFor="dataset-type">SUT lane</label>
                                <select
                                    id="dataset-type"
                                    value={sutType}
                                    onChange={(event) => {
                                        const next = toSutType(event.target.value);
                                        if (next) setSutType(next);
                                    }}
                                >
                                    {SUT_ORDER.map((sut) => (
                                        <option key={sut} value={sut}>
                                            {SUT_INFO[sut].label}
                                        </option>
                                    ))}
                                </select>
                            </div>
                        </div>

                        <div className="form-group">
                            <label htmlFor="dataset-description">Description</label>
                            <textarea
                                id="dataset-description"
                                value={description}
                                onChange={(event) => setDescription(event.target.value)}
                                rows={3}
                                placeholder="Short operator note for this dataset package"
                            />
                        </div>

                        <div className="upload-zone">
                            <Icon name="archive" className="upload-zone-icon" />
                            <div>
                                <strong>{file ? file.name : 'Choose a zip package'}</strong>
                                <p className="text-secondary">
                                    {file ? formatBytes(file.size) : 'The API rejects non-zip uploads.'}
                                </p>
                            </div>
                            <input
                                type="file"
                                accept=".zip"
                                ref={fileInputRef}
                                onChange={(event) => {
                                    const selected = event.target.files?.[0] ?? null;
                                    if (selected && !selected.name.toLowerCase().endsWith('.zip')) {
                                        setFile(null);
                                        setError('Only .zip uploads are supported.');
                                        return;
                                    }
                                    setError(null);
                                    setFile(selected);
                                }}
                            />
                        </div>

                        <button type="submit" className="btn-primary" disabled={uploading}>
                            <Icon name="archive" className="btn-icon" />
                            {uploading ? 'Uploading...' : 'Upload dataset'}
                        </button>
                    </form>
                </div>

                <aside className="panel requirements-panel">
                    <div className="section-header">
                        <div>
                            <h2>{selectedSut.shortLabel} Zip Contract</h2>
                            <p className="text-secondary">{selectedSut.description}</p>
                        </div>
                    </div>
                    <ul className="check-list">
                        {selectedSut.requiredFiles.map((item) => (
                            <li key={item}>
                                <Icon name="check" className="check-list-icon" />
                                <code>{item}</code>
                            </li>
                        ))}
                    </ul>
                </aside>
            </section>

            <section className="panel">
                <div className="section-header">
                    <div>
                        <h2>Dataset Catalog</h2>
                        <p className="text-secondary">Uploaded packages available for retraining.</p>
                    </div>
                    <Link className="btn-secondary" to="/jobs">
                        <Icon name="play" className="btn-icon" />
                        Create job
                    </Link>
                </div>

                {loading ? (
                    <p className="text-secondary block-copy">Loading datasets...</p>
                ) : datasets.length === 0 ? (
                    <p className="text-secondary block-copy">No dataset packages have been uploaded yet.</p>
                ) : (
                    <div className="grid dataset-grid">
                        {datasets.map((dataset) => (
                            <DatasetCard
                                key={dataset.id}
                                dataset={dataset}
                                previewLoading={previewLoadingId === dataset.id}
                                onPreview={handlePreview}
                                onDelete={handleDelete}
                            />
                        ))}
                    </div>
                )}
            </section>

            {preview && (
                <section className="panel">
                    <div className="section-header">
                        <div>
                            <h2>Dataset Preview</h2>
                            <p className="text-secondary">Summary-first view for {preview.name}.</p>
                        </div>
                        <button className="btn-secondary" onClick={() => setPreview(null)}>
                            <Icon name="x" className="btn-icon" />
                            Close
                        </button>
                    </div>

                    <div className="preview-meta">
                        <div>
                            <span className="text-secondary">Total rows</span>
                            <p>{preview.total_rows}</p>
                        </div>
                        <div>
                            <span className="text-secondary">Columns</span>
                            <p>{renderColumns(preview.columns)}</p>
                        </div>
                    </div>

                    {preview.statistics && (
                        <div className="summary-grid">
                            {Object.entries(preview.statistics).map(([key, value]) => (
                                <div key={key} className="summary-card">
                                    <span className="summary-label">{formatKey(key)}</span>
                                    <div className="summary-value">{renderStatValue(value)}</div>
                                </div>
                            ))}
                        </div>
                    )}

                    {preview.rows.length > 0 && (
                        <details className="details-panel">
                            <summary>Show sample rows</summary>
                            <div className="table-wrapper">
                                <table className="data-table">
                                    <thead>
                                        <tr>
                                            {preview.columns.map((col) => (
                                                <th key={col}>{col}</th>
                                            ))}
                                        </tr>
                                    </thead>
                                    <tbody>
                                        {preview.rows.slice(0, 10).map((row, index) => (
                                            <tr key={index}>
                                                {preview.columns.map((col) => (
                                                    <td key={col}>{String(row[col] ?? '')}</td>
                                                ))}
                                            </tr>
                                        ))}
                                    </tbody>
                                </table>
                            </div>
                        </details>
                    )}
                </section>
            )}
        </div>
    );
}

interface DatasetCardProps {
    dataset: Dataset;
    previewLoading: boolean;
    onPreview: (datasetId: number) => void;
    onDelete: (datasetId: number) => void;
}

function DatasetCard({ dataset, previewLoading, onPreview, onDelete }: DatasetCardProps) {
    const sut = toSutType(dataset.sut_type);
    const info = sut ? SUT_INFO[sut] : null;

    return (
        <article className="dataset-card">
            <div className="dataset-header">
                <div>
                    <span className="sut-domain">{info?.domain ?? 'Dataset'}</span>
                    <h3>{dataset.name}</h3>
                </div>
                <span className="tag">{info?.shortLabel ?? dataset.sut_type}</span>
            </div>
            <p className="text-secondary">{dataset.description || info?.description || 'No description'}</p>
            <div className="dataset-meta">
                <span>Rows: {dataset.row_count ?? 'N/A'}</span>
                <span>Size: {formatBytes(dataset.file_size)}</span>
                <span>Created: {formatDate(dataset.created_at)}</span>
            </div>
            <p className="text-secondary">
                File: <span className="dataset-file">{dataset.filename}</span>
            </p>
            <div className="dataset-actions">
                <button className="btn-secondary" onClick={() => onPreview(dataset.id)} disabled={previewLoading}>
                    <Icon name="file" className="btn-icon" />
                    {previewLoading ? 'Loading...' : 'Preview'}
                </button>
                <button className="btn-danger" onClick={() => onDelete(dataset.id)}>
                    <Icon name="x" className="btn-icon" />
                    Delete
                </button>
            </div>
        </article>
    );
}
