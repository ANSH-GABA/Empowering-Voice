import { NavLink, Route, Routes, useLocation } from "react-router-dom";
import { Hand } from "lucide-react";
import LandingPage from "../pages/LandingPage";
import TranslatePage from "../pages/TranslatePage";
import CollectPage from "../pages/CollectPage";

const linkClass = ({ isActive }: { isActive: boolean }) =>
  `rounded-md px-4 py-2 text-lg font-bold ${
    isActive ? "bg-ink text-white" : "text-ink hover:bg-line"
  }`;

export default function App() {
  const { pathname } = useLocation();
  const showAppHeader = pathname !== "/";

  return (
    <div className="min-h-screen">
      {showAppHeader && (
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-line bg-white px-6 py-3">
          <div className="flex items-center gap-2 text-2xl font-bold">
            <Hand className="text-signal" aria-hidden />
            Empowering Voice
          </div>
          <nav className="flex gap-2" aria-label="Main">
            <NavLink to="/translate" className={linkClass}>
              Translate
            </NavLink>
            <NavLink to="/collect" className={linkClass}>
              Collect data
            </NavLink>
            <NavLink to="/" end className={linkClass}>
              Home
            </NavLink>
          </nav>
        </header>
      )}
      <Routes>
        <Route path="/" element={<LandingPage />} />
        <Route path="/translate" element={<main className="mx-auto max-w-7xl p-4 sm:p-6"><TranslatePage /></main>} />
        <Route path="/collect" element={<main className="mx-auto max-w-7xl p-4 sm:p-6"><CollectPage /></main>} />
      </Routes>
    </div>
  );
}
