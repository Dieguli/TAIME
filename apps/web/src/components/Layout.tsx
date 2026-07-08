import { Link, NavLink, Outlet } from 'react-router-dom';

import { Icon, type IconName } from './Icons';

import './Layout.css';

const navItems: { to: string; label: string; icon: IconName; end?: boolean }[] = [
    { to: '/', label: 'Control Center', icon: 'gauge', end: true },
    { to: '/datasets', label: 'Datasets', icon: 'database' },
    { to: '/jobs', label: 'Retraining', icon: 'play' },
    { to: '/models', label: 'Models', icon: 'box' },
];

export function Layout() {
    return (
        <div className="layout">
            <header className="app-header">
                <Link to="/" className="app-header-logo">
                    <img src="/themis-logo.png" alt="THEMIS 5.0" />
                </Link>
            </header>

            <div className="app-body">
                <aside className="sidebar">
                    <div className="sidebar-header">
                        <span className="logo">TAIME</span>
                        <span className="version">v2.0</span>
                    </div>

                    <nav className="nav" aria-label="Primary">
                        {navItems.map((item) => (
                            <NavLink
                                key={item.to}
                                to={item.to}
                                className={({ isActive }) =>
                                    isActive ? 'nav-link active' : 'nav-link'
                                }
                                end={item.end}
                                aria-label={item.label}
                            >
                                <Icon name={item.icon} className="nav-icon" />
                                <span className="nav-text">{item.label}</span>
                            </NavLink>
                        ))}
                    </nav>

                    <div className="sidebar-footer">
                        <a
                            href="/docs"
                            target="_blank"
                            rel="noreferrer"
                            className="nav-link"
                            aria-label="API Docs"
                        >
                            <Icon name="file" className="nav-icon" />
                            <span className="nav-text">API Docs</span>
                        </a>
                    </div>
                </aside>

                <main className="main-content">
                    <Outlet />
                </main>
            </div>

            <footer className="app-footer">
                <div className="footer-section footer-left">
                    <img src="/eu-flag.svg" alt="European Union" height="24" />
                    <span>Funded by the European Union</span>
                </div>
                <div className="footer-section footer-center">
                    THEMIS 5.0 has received funding from the European Union&apos;s Horizon Europe
                    research and innovation programme under grant agreement No 101135049
                </div>
                <div className="footer-section footer-right">
                    <span>Terms of use</span>
                    <span className="footer-sep">|</span>
                    <span>Privacy Policy</span>
                    <span className="footer-sep">|</span>
                    <span>About</span>
                    <span className="footer-sep">|</span>
                    <span>THEMIS 5.0</span>
                </div>
            </footer>
        </div>
    );
}
