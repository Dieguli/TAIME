import type { SVGProps } from 'react';

export type IconName =
    | 'activity'
    | 'archive'
    | 'box'
    | 'check'
    | 'chevron'
    | 'cpu'
    | 'database'
    | 'download'
    | 'file'
    | 'gauge'
    | 'play'
    | 'refresh'
    | 'warning'
    | 'x';

const iconPaths: Record<IconName, string[]> = {
    activity: ['M4 12h4l2-7 4 14 2-7h4'],
    archive: ['M4 7h16', 'M6 7v13h12V7', 'M8 3h8l2 4H6l2-4Z', 'M10 12h4'],
    box: ['M4 8l8-4 8 4-8 4-8-4Z', 'M4 8v8l8 4 8-4V8', 'M12 12v8'],
    check: ['M4 12l5 5L20 6'],
    chevron: ['M8 5l7 7-7 7'],
    cpu: [
        'M8 8h8v8H8V8Z',
        'M4 10h4',
        'M4 14h4',
        'M16 10h4',
        'M16 14h4',
        'M10 4v4',
        'M14 4v4',
        'M10 16v4',
        'M14 16v4',
    ],
    database: ['M5 7c0-2 3-4 7-4s7 2 7 4-3 4-7 4-7-2-7-4Z', 'M5 7v5c0 2 3 4 7 4s7-2 7-4V7', 'M5 12v5c0 2 3 4 7 4s7-2 7-4v-5'],
    download: ['M12 4v10', 'M7 10l5 5 5-5', 'M5 20h14'],
    file: ['M7 3h8l4 4v14H7V3Z', 'M15 3v5h5', 'M9 13h6', 'M9 17h6'],
    gauge: ['M4 14a8 8 0 1 1 16 0', 'M12 14l4-4', 'M7 18h10'],
    play: ['M8 5v14l11-7-11-7Z'],
    refresh: ['M19 8a7 7 0 0 0-12-3L5 7', 'M5 4v3h3', 'M5 16a7 7 0 0 0 12 3l2-2', 'M19 20v-3h-3'],
    warning: ['M12 4 21 20H3L12 4Z', 'M12 9v5', 'M12 17h.01'],
    x: ['M6 6l12 12', 'M18 6 6 18'],
};

interface IconProps extends SVGProps<SVGSVGElement> {
    name: IconName;
    title?: string;
}

export function Icon({ name, title, className, ...props }: IconProps) {
    return (
        <svg
            className={className}
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden={title ? undefined : true}
            role={title ? 'img' : undefined}
            {...props}
        >
            {title && <title>{title}</title>}
            {iconPaths[name].map((path, index) => (
                <path key={`${name}-${index}`} d={path} />
            ))}
        </svg>
    );
}
