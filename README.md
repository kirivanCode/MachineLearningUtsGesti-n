# Optimizador de Horarios UTS

Sistema de generacíon de clases UTS basado en Python que utiliza machine learning para optimizar horarios. Considera múltiples restricciones incluyendo disponibilidad de profesores, capacidad de aulas y requisitos estudiantiles.

## Características

- Optimización de horarios basada en machine learning
- Aprendizaje adaptativo de patrones anteriores
- Generación de horarios en tiempo real
- Optimización multicriterio para asignación de recursos
- Exportación a Excel
- Dashboard interactivo con analíticas
- Integración con API REST

## Arquitectura

El sistema consta de tres componentes principales:

### 1. AgenteAdaptativo
Maneja el aprendizaje dinámico y ajuste de parámetros:
- Mantiene historial de rendimiento
- Actualiza parámetros automáticamente
- Guarda y carga estado entre sesiones
- Sugiere parámetros basados en rendimiento histórico

### 2. OptimizadorHorarios
Motor principal de programación que:
- Gestiona generación de horarios
- Maneja asignación de recursos
- Integra modelos de machine learning
- Procesa restricciones y requerimientos
- Mantiene historial de programación

### 3. Interfaz Streamlit
Interfaz de usuario que provee:
- Dashboard interactivo
- Gestión de configuración
- Interfaz de entrenamiento
- Generación de horarios
- Analíticas e informes

## Detalles Técnicos

### Dependencias
```python
streamlit
pandas
numpy
scikit-learn
plotly
openpyxl
requests
faker
joblib
```

### Modelos de Datos

#### Horario de Clase
```python
{
    'grupo': str,            # Identificador de grupo
    'dia_semana': str,      # Día de la semana
    'hora_inicio': str,     # Hora inicio
    'hora_fin': str,        # Hora fin
    'alumnos': int,         # Cantidad de estudiantes
    'materia_id': int,      # ID de materia
    'salon_id': int,        # ID de salón
    'profesor_id': int      # ID de profesor
}
```

### Algoritmos Principales

#### Generación de Horarios
1. División de Grupos
   - Divide clases grandes en grupos manejables
   - Asigna identificadores únicos
   - Considera restricciones de capacidad

2. Asignación de Horarios
   - Distribuye clases en días disponibles
   - Mantiene límites de carga docente
   - Respeta disponibilidad de aulas
   - Sigue restricciones de bloques

3. Optimización de Recursos
   - Balancea carga docente
   - Maximiza utilización de aulas
   - Minimiza conflictos
   - Optimiza distribución estudiantil

### Componentes de Machine Learning

#### Modelos Soportados
- Random Forest Classifier
- K-Nearest Neighbors Classifier

#### Características Utilizadas
```python
[
    'experiencia',           # Experiencia docente
    'calificacion_alumno',   # Calificación estudiantes
    'alumnos',              # Número de estudiantes
    'bloques',              # Bloques necesarios
    'horarios_disponibles',  # Horarios disponibles
    'capacidad_salon',      # Capacidad del aula
    'conflictos_horario',   # Conflictos horarios
    'carga_profesor'        # Carga docente
]
```

### Gestión de Configuración

#### Parámetros Básicos
- Mínimo de estudiantes por clase
- Máxima carga docente
- Duración de bloques
- Días laborables
- Límites de horario

#### Configuración Avanzada
- Nivel de optimización
- Máximo de iteraciones
- Detección de patrones
- Ajustes de auto-corrección

#### Restricciones
- Límite de clases consecutivas
- Tiempo mínimo de descanso
- Máximo de ventanas por día
- Restricciones de distancia

## Integración API

### Endpoints Utilizados
```
GET /api/profesores
GET /api/materias
GET /api/salones
GET /api/horarios_disponibles
GET /api/profesor_materia
POST /api/clases
```

### Flujo de Datos
1. Obtiene datos maestros de API
2. Procesa y optimiza horarios
3. Envía horarios generados
4. Mantiene sincronización

## Uso

### Instalación
```bash
pip install -r requirements.txt
```

### Ejecución
```bash
streamlit run ending.py o como lo llamó
```

### Configuración
1. Acceder a pestaña Configuración
2. Establecer parámetros básicos
3. Configurar ajustes avanzados
4. Definir restricciones
5. Guardar configuración

### Generación de Horarios
1. Entrenar modelo con datos históricos
2. Configurar parámetros
3. Generar horarios
4. Revisar y exportar resultados

## Mejores Prácticas

### Optimización de Rendimiento
- Procesamiento por lotes
- Caché de datos frecuentes
- Optimización de consultas
- Indexación apropiada

### Manejo de Errores
- Captura completa de errores
- Mensajes claros
- Registro para depuración
- Consistencia de datos

### Gestión de Datos
- Respaldos regulares
- Control de versiones
- Validación de datos
- Limpieza de temporales

## Licencia

Este proyecto está bajo la Licencia de Ivan Sierra y Jhoan Sebastian Jimenez 
