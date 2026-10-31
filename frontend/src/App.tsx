import { useState } from 'react'
import DetectionView from './DetectionView'
import TutorialView from './TutorialView'

type Tab = 'home' | 'detect' | 'about'

// Menampilkan halaman aplikasi dan mempertahankan status deteksi saat tab berubah.
export default function App() {
  const [tab, setTab] = useState<Tab>('home')
  const [open, setOpen] = useState(window.innerWidth > 700)

  // Berpindah halaman dan menutup menu pada layar ponsel.
  function go(t: Tab) {
    setTab(t)
    if (window.innerWidth <= 700) setOpen(false)
  }

  return (
    <div className="layout">
      <header>
        <span className="burger" onClick={() => setOpen(!open)}>
          ☰
        </span>
        <div className="brand" onClick={() => go('home')}>
          <span className="logo">🍪</span>
          <span className="brand-name">Deteksi Snack</span>
        </div>
      </header>

      <div className="body">
        {open && (
          <nav>
            <button
              className={tab === 'detect' ? 'active' : ''}
              onClick={() => go('detect')}
            >
              Product Detection
            </button>
            <button
              className={tab === 'about' ? 'active' : ''}
              onClick={() => go('about')}
            >
              Tutorial & About
            </button>
          </nav>
        )}

        <main>
          {tab === 'home' && (
            <div className="home">
              <h1>
                SNACK
                <br />
                DETECTION
              </h1>
              <h2>
                KALORI
                <br />
                TERENDAH
              </h2>
              <button className="primary" onClick={() => go('detect')}>
                Mulai Deteksi
              </button>
            </div>
          )}
          {tab === 'detect' && (
            <h2 className="page-title">Product Detection</h2>
          )}
          {tab === 'about' && (
            <h2 className="page-title">Tutorial & About</h2>
          )}
          <div style={{ display: tab === 'detect' ? 'block' : 'none' }}>
            <DetectionView />
          </div>
          {tab === 'about' && <TutorialView />}
        </main>
      </div>
    </div>
  )
}
