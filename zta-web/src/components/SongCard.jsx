export default function SongCard({ song, distanceLabel = 'distancia', highlight = false }) {
  return (
    <li className={`song ${highlight ? 'shared' : ''}`}>
      <span className="song-rank">{song.rank}</span>
      <div className="song-main">
        <strong title={song.track_name}>{song.track_name}</strong>
        <span className="muted">{song.artists}</span>
        <div className="chips small-chips">
          {song.genres.slice(0, 4).map((g) => <span key={g} className="chip">{g}</span>)}
        </div>
      </div>
      <div className="song-stats">
        <div title="Popularidad registrada (0-100)">
          <span className="muted small">Popularidad</span>
          <div className="bar thin"><div style={{ width: `${Math.max(2, song.popularity)}%` }} /></div>
          <span className="small">{Math.round(song.popularity)}</span>
        </div>
        <div className="song-distance" title={`${distanceLabel} al perfil`}>
          <span className="muted small">{distanceLabel}</span>
          <strong>{song.distance.toFixed(3)}</strong>
        </div>
      </div>
    </li>
  )
}
