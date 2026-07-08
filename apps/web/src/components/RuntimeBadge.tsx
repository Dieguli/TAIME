import type { RuntimeInfo } from '../api/client';
import { Icon } from './Icons';

interface RuntimeBadgeProps {
    runtime?: RuntimeInfo | null;
    compact?: boolean;
}

export function RuntimeBadge({ runtime, compact = false }: RuntimeBadgeProps) {
    const available = Boolean(runtime?.cuda_available);
    const label = available ? 'GPU ready' : 'CPU runtime';
    const detail = available
        ? runtime?.device_name ?? 'CUDA device available'
        : runtime?.image === 'gpu'
          ? 'GPU image, CUDA not visible'
          : 'CUDA not available';

    return (
        <div className={`runtime-badge ${available ? 'runtime-badge-ready' : 'runtime-badge-cpu'}`}>
            <Icon name="cpu" className="runtime-badge-icon" />
            <div>
                <span>{label}</span>
                {!compact && <small>{detail}</small>}
            </div>
        </div>
    );
}
