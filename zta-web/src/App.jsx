import { useEffect, useState } from 'react'
import { AppProvider } from './state.jsx'
import Navbar, { ROUTES } from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import ProfilePage from './pages/ProfilePage.jsx'
import RecommendPage from './pages/RecommendPage.jsx'
import ModelsPage from './pages/ModelsPage.jsx'
import LabPage from './pages/LabPage.jsx'

const PAGES = { '': Home, perfil: ProfilePage, recomendar: RecommendPage, modelos: ModelsPage, laboratorio: LabPage }

function useHashRoute() {
  const read = () => {
    const key = window.location.hash.replace(/^#\/?/, '')
    return ROUTES.some((r) => r.key === key) ? key : ''
  }
  const [route, setRoute] = useState(read)
  useEffect(() => {
    const onChange = () => {
      setRoute(read())
      window.scrollTo({ top: 0 })
    }
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return route
}

export default function App() {
  const route = useHashRoute()
  const Page = PAGES[route]
  return (
    <AppProvider>
      <Navbar route={route} />
      <main className="container">
        <Page />
      </main>
      <footer className="footer">
        <span>ZTA Music · Proyecto académico · Universidad Surcolombiana</span>
        <span className="muted">Demostración: no requiere cuenta y no guarda datos personales.</span>
      </footer>
    </AppProvider>
  )
}
