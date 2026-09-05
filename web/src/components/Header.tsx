/* Top bar: wordmark and the two primary tabs. */

export type Tab = "yields" | "prices";

interface Props {
  tab: Tab;
  onTab: (tab: Tab) => void;
}

const TABS: { id: Tab; label: string }[] = [
  { id: "yields", label: "Yields" },
  { id: "prices", label: "Prices" },
];

export function Header({ tab, onTab }: Props) {
  return (
    <header className="header">
      <div className="wordmark">
        Deep<span>Harvest</span>
      </div>

      <nav className="tabs" aria-label="Sections">
        {TABS.map((t) => (
          <button
            key={t.id}
            type="button"
            className={tab === t.id ? "tab active" : "tab"}
            aria-current={tab === t.id ? "page" : undefined}
            onClick={() => onTab(t.id)}
          >
            {t.label}
          </button>
        ))}
      </nav>

      <p className="header-note">County-level US crop analytics</p>
    </header>
  );
}
