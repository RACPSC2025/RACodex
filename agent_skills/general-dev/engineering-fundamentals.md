# 📐 Engineering Fundamentals

> Conocimiento agnóstico al lenguaje que todo Senior Engineer debe dominar.
> Es la base que hace que las decisiones técnicas tengan fundamento.

---

## I. DSA — Estructuras de Datos y Algoritmos

### Complejidad — la brújula de todo Senior Engineer

| Complejidad | Nombre | Ejemplo concreto |
|-------------|--------|-----------------|
| O(1) | Constante | Acceso a dict/hash, push stack |
| O(log n) | Logarítmica | Binary search, árbol balanceado |
| O(n) | Lineal | Recorrer lista, búsqueda lineal |
| O(n log n) | Lineal-logarítmica | Merge sort, Tim sort (Python default) |
| O(n²) | Cuadrática | Bubble sort, loops anidados naïve |
| O(2ⁿ) | Exponencial | Fibonacci recursivo sin memoización |
| O(n!) | Factorial | Permutaciones brutas |

**Regla práctica para decisiones técnicas:**
- n < 1,000 → casi cualquier solución funciona
- n < 100,000 → O(n log n) o mejor
- n < 10,000,000 → O(n) o mejor
- n > 10,000,000 → necesitas O(log n) o O(1)

### Estructuras de datos — cuándo usar cada una

| Estructura | Acceso | Búsqueda | Insert | Delete | Cuándo usar |
|-----------|--------|----------|--------|--------|-------------|
| **Array/Lista** | O(1) | O(n) | O(1) amortizado | O(n) | Acceso por índice, tamaño fijo |
| **Hash Map/Dict** | O(1) | O(1) | O(1) | O(1) | Búsqueda por clave, contar frecuencias |
| **Set** | O(1) | O(1) | O(1) | O(1) | Unicidad, membership checks |
| **Linked List** | O(n) | O(n) | O(1) | O(1) | Inserciones frecuentes al inicio, LRU cache |
| **Stack (LIFO)** | — | — | O(1) | O(1) | Undo/redo, DFS, parsing de expresiones |
| **Queue/Deque** | — | — | O(1) | O(1) | BFS, task queues, sliding window |
| **Heap/Priority Queue** | O(log n) | O(log n) | O(log n) | O(log n) | k elementos más grandes, scheduling, Dijkstra |
| **Tree (BST, AVL)** | O(log n) | O(log n) | O(log n) | O(log n) | Datos ordenados, range queries, indexes |
| **Trie** | O(m) | O(m) | O(m) | O(m) | Autocomplete, spell checker, prefix matching |
| **Graph** | — | O(V+E) | O(1) | O(V+E) | Relaciones complejas, pathfinding, redes |

### Algoritmos esenciales para decisiones técnicas

- **Binary Search** — O(log n), requiere datos ordenados. Aplicar en: búsqueda con índice, feature flags, version ranges.
- **Two Pointers** — O(n), arrays ordenados. Aplicar en: par con suma, palíndromos, merge de sorted arrays.
- **Sliding Window** — O(n), subarrays. Aplicar en: rate limiting, análisis de logs en ventana de tiempo.
- **BFS** — O(V+E), nivel por nivel. Aplicar en: camino más corto, relaciones sociales, permisos heredados.
- **DFS** — O(V+E), profundidad. Aplicar en: detección de ciclos, topological sort, componentes conectados.
- **Dijkstra** — O((V+E) log V), grafos ponderados positivos. Aplicar en: routing, costos mínimos.
- **Dynamic Programming** — memoización o tabulación. Aplicar en: pricing con descuentos, pathfinding, sequence matching.

---

## II. Patrones de Diseño

### Principios SOLID

| Principio | Definición práctica | Violación común |
|-----------|--------------------|-----------------|
| **S** — Single Responsibility | Una clase/módulo tiene una sola razón para cambiar | God class con 500 líneas y 20 métodos |
| **O** — Open/Closed | Abierto a extensión, cerrado a modificación | `if tipo == "A": ... elif tipo == "B":` infinito |
| **L** — Liskov Substitution | Subclase reemplaza a clase padre sin romper nada | Subclase que lanza excepciones en métodos heredados |
| **I** — Interface Segregation | Interfaces pequeñas y específicas | Interfaz con 15 métodos donde cada clase implementa 3 |
| **D** — Dependency Inversion | Depender de abstracciones, no de implementaciones | `service = MySQLRepository()` hardcodeado |

### Patrones Creacionales

| Patrón | Cuándo | Ejemplo |
|--------|--------|---------|
| **Factory Method** | Múltiples tipos con interfaz común | NotificationFactory → Email, Slack, SMS |
| **Abstract Factory** | Familias de objetos relacionados | UI components: Web vs Mobile vs CLI |
| **Builder** | Construcción paso a paso | QueryBuilder, EmailBuilder, ReportBuilder |
| **Singleton** | Una sola instancia global (cuidado con testing) | Connection pool, config global, registry |

### Patrones Estructurales

| Patrón | Cuándo | Ejemplo |
|--------|--------|---------|
| **Adapter** | Interfaces incompatibles | Adaptar SDK externo a tu interfaz interna |
| **Decorator** | Añadir comportamiento sin modificar clase | @cache, @retry, @require_permission |
| **Facade** | Interfaz simplificada sobre sistema complejo | OrderService sobre Inventory + Payment + Shipping |
| **Proxy** | Control de acceso a objeto | Lazy loading, logging, control de permisos |
| **Composite** | Tratar individuales y colecciones uniformemente | Estructuras en árbol, permisos jerárquicos |

### Patrones de Comportamiento

| Patrón | Cuándo | Ejemplo |
|--------|--------|---------|
| **Strategy** | Familia de algoritmos intercambiables | Algoritmos de pricing, métodos de autenticación |
| **Observer/Event** | Notificar cambios a múltiples listeners | Event bus, webhooks |
| **Command** | Encapsular acción como objeto | Celery tasks, undo/redo, audit logs |
| **Chain of Responsibility** | Request por cadena de handlers | Middleware, pipelines de validación |
| **State** | Máquina de estados finita | Order: pending → confirmed → shipped |
| **Template Method** | Esqueleto de algoritmo con pasos override | Importador: parse → validate → transform → save |

### Patrones Modernos de Backend

| Patrón | Descripción |
|--------|------------|
| **Repository** | Abstrae acceso a datos. Interface en dominio, implementación en infraestructura |
| **Unit of Work** | Agrupa operaciones en transacción lógica. SQLAlchemy Session es UoW |
| **CQRS** | Separa commands (writes) de queries (reads). Optimizar independientemente |
| **Saga Pattern** | Transacciones distribuidas sin 2PC. Choreography (eventos) vs Orchestration (orquestador central) |

---

## III. Arquitecturas Backend

### Cuándo usar qué

| Arquitectura | Complejidad | Cuándo aplicar |
|-------------|-------------|----------------|
| **Monolito modular** | Baja | Startup, equipo < 10, dominio no definido |
| **Monolito + async workers** | Media | Escala media, operaciones async necesarias |
| **Microservicios** | Alta | Equipos independientes, escala diferenciada |
| **Event-driven** | Alta | Alta desacoplamiento, procesamiento asíncrono masivo |
| **Serverless** | Media | Workloads esporádicos, sin estado |

**Regla de oro:** Empieza con monolito modular bien estructurado. Extrae microservicios solo cuando tengas razón concreta medible (escala, equipo independiente, deploy independiente).

### Arquitectura Hexagonal (Ports & Adapters)

```
         ┌─────────────────────────────────┐
  HTTP → │  API Adapter (FastAPI)          │
  CLI  → │  CLI Adapter                    │
         │         ↓                       │
         │  ┌─────────────────────┐        │
         │  │   APPLICATION       │        │
         │  │   (Use Cases)       │        │
         │  └─────────────────────┘        │
         │         ↓                       │
         │  ┌─────────────────────┐        │
         │  │     DOMAIN          │        │
         │  │  (Entities, Rules)  │        │
         │  └─────────────────────┘        │
         │         ↓                       │
         │  DB Adapter → DB               │
         │  Cache Adapter → Redis         │
         └─────────────────────────────────┘
```

**Reglas de dependencia:**
- El dominio no importa nada de infraestructura
- Los adapters dependen del dominio, nunca al revés
- Los ports (interfaces) se definen en el dominio o aplicación
- Los adapters (implementaciones) viven en infraestructura

### Microservicios — Checklist de extracción

- [ ] ¿El equipo responsable es independiente?
- [ ] ¿Necesita escalar de forma diferente al resto?
- [ ] ¿Tiene un dominio claramente delimitado (bounded context)?
- [ ] ¿El costo de complejidad distribuida < beneficio?

Si respondiste NO a 2 o más → no extraigas todavía.

### Patrones de Microservicios

| Patrón | Qué resuelve |
|--------|-------------|
| **API Gateway** | Punto de entrada único: auth, rate limiting, routing |
| **Service Discovery** | Cómo se encuentran los servicios (client-side vs server-side) |
| **Circuit Breaker** | Prevenir cascada de fallos: Closed → Open → Half-Open |
| **Distributed Tracing** | Propagar trace_id entre servicios (OpenTelemetry, Jaeger) |

### Event-Driven Architecture

```
Producer → Event Broker → Consumer(s)
              ↓
        Kafka / RabbitMQ / Redis Streams
```

| Concepto | Descripción |
|----------|------------|
| **Event** | "UserRegistered" → pasó, informamos, múltiples consumidores |
| **Command** | "SendWelcomeEmail" → solicitud de acción, un ejecutor |
| **Query** | "GetUserById" → solicitud de datos, respuesta esperada |
| **At-least-once** | Puede llegar duplicado → consumidores idempotentes |
| **At-most-once** | Puede perderse → para métricas no críticas |

**Kafka vs RabbitMQ vs Redis Streams:**
- **Kafka**: Retención días/semanas, throughput muy alto, complejidad alta. Para event sourcing, audit.
- **RabbitMQ**: Retención hasta consumo, alto throughput, complejidad media. Para task queues, routing complejo.
- **Redis Streams**: Retención configurable, alto throughput, complejidad baja. Para cuando ya tienes Redis.

### Event Sourcing

Estado actual = fold de todos los eventos históricos.

**Cuándo usar:** Audit log completo requerido, reconstruir estado en cualquier punto del tiempo, múltiples proyecciones (CQRS natural).

**Cuándo NO usar:** La mayoría de apps CRUD estándar — complejidad no justificada.

---

## IV. System Design — Marco de Decisiones

### Framework de System Design

1. **Clarificar requerimientos** — Funcionales, no funcionales, escala, constraints
2. **Estimaciones de escala** — Usuarios × req/usuario/día = RPS. Storage: tamaño × objetos × días
3. **Diseño de alto nivel** — API Layer → Application Layer → Data Layer
4. **Resolver componentes críticos** — HA, DB scaling, cache, consistencia

### CAP Theorem

| Sistema | Elige | Por qué |
|---------|-------|---------|
| Banco, pagos | CP | Consistencia crítica sobre disponibilidad |
| DNS | AP | Disponibilidad crítica, eventual consistency OK |
| Red social (likes) | AP | Eventual consistency aceptable |
| Inventario e-commerce | CP o AP | Depende del negocio |

### Escalado

| Estrategia | Descripción | Cuándo |
|-----------|------------|--------|
| **Vertical** | Más CPU/RAM | Rápido pero single point of failure |
| **Horizontal** | Más servidores | Requiere stateless apps, load balancer |
| **Read replicas** | Escalar reads (80-90% del tráfico) | Read-heavy workloads |
| **Caching** | Eliminar reads repetidas | Datos leídos frecuentemente, pocos writes |
| **Partitioning** | Dividir tabla por rango/hash | Tablas muy grandes, queries predecibles |
| **Sharding** | Dividir datos entre múltiples DBs | Escala extrema, complejidad alta |

### Cache Strategies

| Estrategia | Descripción | Riesgo |
|-----------|------------|--------|
| **Cache-aside** | App consulta cache → si miss → consulta DB → guarda en cache | Cache miss frecuente al inicio |
| **Write-through** | App escribe DB + cache simultáneamente | Latencia de write más alta |
| **Write-behind** | App escribe cache → cache escribe DB async | Riesgo de pérdida de datos |

**Cache stampede:** muchos requests simultáneos hacen miss, todos van a DB.
Solución: mutex lock en cache, probabilistic early expiration, background refresh.

### Decisiones de Tech Lead — 5 preguntas clave

1. ¿Cuál es el problema específico que esto resuelve? ¿Tenemos ese problema ahora?
2. ¿Cuál es el costo de operación y mantenimiento?
3. ¿El equipo puede operarlo en producción a las 3am si falla?
4. ¿Cuál es el plan de migración si esto no funciona?
5. ¿Qué pasa con los datos si el servicio cae?

---

## V. Bases de Datos

### Relacionales (SQL)

**Usar cuando:** Datos con relaciones definidas, ACID necesario, queries ad-hoc frecuentes, esquema estable.

**PostgreSQL preferido para backend Python:**
- JSONB nativo con índices GIN
- Arrays, tipos personalizados
- Window functions completas
- Full-text search integrado

**Reglas de diseño:**
- Normaliza hasta 3NF como punto de partida
- Desnormaliza solo con evidencia de query lenta (EXPLAIN ANALYZE primero)
- Siempre: PK surrogate (UUID o BIGSERIAL), created_at/updated_at
- Índices: columnas en WHERE, JOIN, ORDER BY frecuentes
- Nunca: índice en columna de baja cardinalidad

### NoSQL — Cuándo y cuál

| Tipo | Ejemplo | Cuándo usar | Evitar cuando |
|------|---------|-------------|--------------|
| **Document** | MongoDB, Firestore | Datos semiestructurados, esquema cambiante | Joins frecuentes, consistencia fuerte |
| **Key-Value** | Redis, DynamoDB | Caching, sesiones, rate limiting | Queries complejas, datos relacionales |
| **Wide-Column** | Cassandra, ScyllaDB | Time-series, write-heavy, distribución global | Queries ad-hoc, joins |
| **Search** | Elasticsearch | Full-text search, faceted search, logs | DB primaria, consistencia fuerte |
| **Graph** | Neo4j | Relaciones complejas son el dominio | Relaciones simples FK |

---

## VI. DevOps y Cultura de Ingeniería

### Los 4 métricas DORA

| Métrica | Élite | Alta | Media | Baja |
|---------|-------|------|-------|------|
| Deployment Frequency | Múltiples/día | 1/semana | 1/mes | < 1/mes |
| Lead Time for Changes | < 1 hora | 1 día - 1 semana | 1 semana - 1 mes | > 1 mes |
| Change Failure Rate | < 5% | 5-10% | 10-15% | > 15% |
| Time to Restore Service | < 1 hora | < 1 día | < 1 semana | > 1 semana |

### Principios de Cultura de Ingeniería

- **Documenta las decisiones** — ADRs (Architecture Decision Records) para decisiones que tienen consecuencias a largo plazo
- **Automatiza todo** — CI/CD, tests, linting, deploy. Si se hace más de una vez, debe ser automático
- **Fail fast, recover faster** — Mejor detectar el error en CI que en producción a las 3am
- **Code review no es gatekeeping** — Es transferencia de conocimiento y detección temprana de problemas
- **Mide, no adivines** — Métricas > opiniones. Dashboards > suposiciones

---

*Documento de referencia técnica. No es un archivo de instrucciones operativas. Para instrucciones de agente, ver `agent.md`.*
