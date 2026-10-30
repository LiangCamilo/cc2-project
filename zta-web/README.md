# ZTA Music · aplicación web de demostración

Cliente React que consume la API de Python (`python -m zta.api`). No pide inicio de sesión ni guarda datos:
es un demo para probar todas las funcionalidades del recomendador con una interfaz estilo Spotify.

## Qué incluye

| Página | Qué hace | Endpoints que usa |
|---|---|---|
| **Inicio** | Explica el proyecto, muestra cifras del catálogo y los tres modos de escucha. | `GET /health`, `GET /meta` |
| **Perfil** (función 1) | Sliders con interruptor por característica, atajos (Fiesta, Gimnasio…) o perfil a partir de canciones. Muestra radar del perfil, valores reales, popularidad esperada y géneros por modelo con consenso. | `POST /profile`, `GET /songs/search` |
| **Recomendar** (función 2) | Método ZTA, KNN o comparación; modo Familiar/Discovery/Exploration; distancia; penalización por popularidad (λ); cantidad; filtro de géneros; ocultar repetidas. Listas y gráfica de popularidad frente a distancia. | `POST /recommend` |
| **Modelos y métricas** | Resultados del informe (géneros, popularidad, calidad de ZTA) con gráficas y tablas. | – (datos fijos en `src/lib/metrics.js`) |
| **Laboratorio API** | Prueba cada endpoint con JSON editable, ve la respuesta cruda, el tiempo y el historial de todas las llamadas. | todos |

## Cómo correrla

Requisitos: Node 20.19 o superior (probado con Node 22) y la API de Python funcionando.

```bash
# 1) En la raíz del proyecto Python, con el entorno virtual activo:
python -m zta.api --port 8000

# 2) En otra terminal, dentro de esta carpeta:
npm install
npm run dev          # abre http://localhost:5173
```

`npm run build` genera la versión de producción en `dist/` y `npm run preview` la sirve en el puerto 4173.

## Cómo se conecta a la API

El navegador llama a `/api/...` y Vite reenvía esas peticiones a `http://127.0.0.1:8000` (ver `vite.config.js`),
por lo que **no hace falta configurar CORS** en desarrollo.

| Situación | Qué hacer |
|---|---|
| La API corre en otro puerto o equipo | `VITE_PROXY_TARGET=http://host:puerto npm run dev` |
| Publicar la web separada de la API | Compila con `VITE_API_URL=https://tu-api npm run build` y arranca la API con `ZTA_CORS_ORIGINS=https://tu-web` |

Si la API está apagada, la barra superior muestra «API sin conexión» y cada página ofrece reintentar.

## Estructura

```
src/
  lib/api.js          cliente HTTP (registra cada llamada para el Laboratorio)
  lib/features.js     nombres, unidades, atajos y modos
  lib/metrics.js      resultados oficiales del informe (editar si se reentrena)
  lib/charts.js       Chart.js con tema oscuro
  state.jsx           perfil compartido entre Perfil y Recomendar
  pages/              Home, ProfilePage, RecommendPage, ModelsPage, LabPage
  components/         Navbar, SliderControl, ProfileBuilder, ProfileResult, SongCard, SongsChart, Ui
  index.css           estilo (negro y verde #1DB954)
```

## Notas

- Las cifras de «Modelos y métricas» son datos fijos copiados del informe (secciones 6.15 y 7.3); no las calcula la API.
- La penalización λ solo afecta a ZTA. El filtro de géneros es una función de la aplicación y cambia las condiciones del experimento del informe.
- Es un cliente de demostración. El servidor Spring Boot deberá consumir la misma API desde el servidor.
