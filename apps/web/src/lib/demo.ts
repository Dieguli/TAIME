import type { Dataset, Job, Model } from '../api/client';

export type SutType = 'healthcare_pc' | 'infra_port' | 'disinfo_fake' | 'disinfo_hate';

export const SUT_ORDER: SutType[] = [
    'healthcare_pc',
    'infra_port',
    'disinfo_fake',
    'disinfo_hate',
];

export const SUT_INFO: Record<
    SutType,
    {
        label: string;
        shortLabel: string;
        domain: string;
        model: string;
        description: string;
        requiredFiles: string[];
        fastPreset: string;
    }
> = {
    healthcare_pc: {
        label: 'Healthcare - PC Risk Predictor',
        shortLabel: 'Healthcare',
        domain: 'Pancreatic cancer risk',
        model: 'XGBoost',
        description: 'Curated MUP MAG package for risk-model retraining.',
        requiredFiles: ['mup_risk_model_data.csv', 'y_train_model_data.csv'],
        fastPreset: '120 estimators, full curated package',
    },
    infra_port: {
        label: 'Infrastructure - Port Operations',
        shortLabel: 'Port operations',
        domain: 'Critical infrastructure',
        model: 'Darts TSMixer',
        description: 'Time-series package with seeded TSMixer artifacts.',
        requiredFiles: [
            'series_meta.json',
            'past_covs_meta.json',
            'series_*.parquet',
            'past_covs_*.parquet',
            '*.pt',
            '*.ckpt',
        ],
        fastPreset: '10 epochs, batch 64, scheduler enabled',
    },
    disinfo_fake: {
        label: 'Disinformation - Fake News',
        shortLabel: 'Fake news',
        domain: 'Disinformation',
        model: 'DistilBERT',
        description: 'Balanced fake/true news package for a fast transformer demo.',
        requiredFiles: ['Fake.csv', 'True.csv'],
        fastPreset: '3 epochs, 2,000-row demo sample',
    },
    disinfo_hate: {
        label: 'Disinformation - Hate Speech',
        shortLabel: 'Hate speech',
        domain: 'Disinformation',
        model: 'RoBERTa',
        description: 'Hate-speech package with the bundled transformer model zip.',
        requiredFiles: [
            'en_dataset.csv',
            'Multitarget-CONAN.csv',
            'atc-code-transformers-hate-speech-detection-*.zip',
            'FakeNews_HateSpeech_model.zip',
        ],
        fastPreset: '5 epochs, binary mode, 2,000-row demo sample',
    },
};

export const toSutType = (value: string): SutType | null =>
    SUT_ORDER.includes(value as SutType) ? (value as SutType) : null;

export const sutLabel = (value: string) => {
    const sut = toSutType(value);
    return sut ? SUT_INFO[sut].label : value.replace(/_/g, ' ');
};

export const formatBytes = (bytes?: number) => {
    if (bytes === undefined || bytes === null) return 'N/A';
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const index = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / Math.pow(1024, index);
    return `${value.toFixed(value >= 10 || index === 0 ? 0 : 1)} ${units[index]}`;
};

export const formatDate = (value?: string) => (value ? new Date(value).toLocaleString() : 'N/A');

export const formatPercent = (value?: number) =>
    value !== undefined && value !== null ? `${value.toFixed(0)}%` : 'N/A';

export const formatMetric = (value: unknown) => {
    if (value === null || value === undefined) return 'N/A';
    if (typeof value === 'number') {
        return Number.isInteger(value) ? value.toString() : value.toFixed(4);
    }
    return String(value);
};

export const formatKey = (value: string) =>
    value.replace(/_/g, ' ').replace(/\b\w/g, (letter) => letter.toUpperCase());

export const statusBadgeClass = (status: Job['status']) => {
    switch (status) {
        case 'pending':
            return 'badge badge-pending';
        case 'running':
            return 'badge badge-running';
        case 'completed':
            return 'badge badge-completed';
        case 'failed':
            return 'badge badge-failed';
        case 'cancelled':
            return 'badge badge-cancelled';
        default:
            return 'badge';
    }
};

export const latestBySut = <T extends { sut_type: string; created_at: string; id: number }>(
    items: T[]
) => {
    const map = new Map<SutType, T>();
    items.forEach((item) => {
        const sut = toSutType(item.sut_type);
        if (!sut) return;
        const current = map.get(sut);
        if (
            !current ||
            new Date(item.created_at).getTime() > new Date(current.created_at).getTime() ||
            item.id > current.id
        ) {
            map.set(sut, item);
        }
    });
    return map;
};

export const latestDatasetBySut = (datasets: Dataset[]) => latestBySut(datasets);
export const latestJobBySut = (jobs: Job[]) => latestBySut(jobs);
export const latestModelBySut = (models: Model[]) => latestBySut(models);

export const metricHighlights = (metrics?: Record<string, unknown> | null) => {
    if (!metrics) return [];
    const preferred = [
        'accuracy',
        'f1_macro',
        'f1',
        'precision_macro',
        'precision',
        'recall_macro',
        'recall',
        'rmse',
        'mae',
        'mape',
        'val_loss',
    ];
    const highlights = preferred
        .filter((key) => metrics[key] !== undefined && metrics[key] !== null)
        .map((key) => ({ key, value: metrics[key] }));

    if (highlights.length > 0) return highlights.slice(0, 4);

    return Object.entries(metrics)
        .filter(([, value]) => typeof value === 'number' || typeof value === 'string')
        .slice(0, 4)
        .map(([key, value]) => ({ key, value }));
};
