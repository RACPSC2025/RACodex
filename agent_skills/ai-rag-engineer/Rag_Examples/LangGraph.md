# Descripción general de LangGraph

LangGraph, en el que confían empresas que están dando forma al futuro de los agentes —entre ellas Klarna, Uber, JP Morgan y otras—, es un marco de orquestación de bajo nivel y un entorno de ejecución para crear, gestionar e implementar agentes con estado y de larga duración.

LangGraph es de muy bajo nivel y se centra exclusivamente en la orquestación de agentes. Antes de usar LangGraph, le recomendamos familiarizarse con algunos de los componentes utilizados para crear agentes, comenzando por los modelos y las herramientas.

En la documentación, utilizaremos habitualmente componentes de LangChain para integrar modelos y herramientas, pero no es necesario usar LangChain para usar LangGraph. Si está empezando a trabajar con agentes o desea una abstracción de nivel superior, le recomendamos que utilice los agentes de LangChain, que proporcionan arquitecturas predefinidas para bucles comunes de LLM y de llamada a herramientas.

LangGraph se centra en las capacidades subyacentes importantes para la orquestación de agentes: ejecución duradera, transmisión de datos, intervención humana y más.

## Instalación

```bash
pip install -U langgraph
```

A continuación, crea un ejemplo sencillo de "hola mundo":

```python
from langgraph.graph import StateGraph, MessagesState, START, END

def mock_llm(state: MessagesState):
    return {"messages": [{"role": "ai", "content": "hello world"}]}

graph = StateGraph(MessagesState)
graph.add_node(mock_llm)
graph.add_edge(START, "mock_llm")
graph.add_edge("mock_llm", END)
graph = graph.compile()

graph.invoke({"messages": [{"role": "user", "content": "hi!"}]})
```

Utilice LangSmith para rastrear solicitudes, depurar el comportamiento del agente y evaluar los resultados. Configure `LANGSMITH_TRACING=true` y su clave API para comenzar.

## Beneficios principales

LangGraph proporciona una infraestructura de soporte de bajo nivel para cualquier flujo de trabajo o agente con estado y de larga duración. LangGraph no abstrae las indicaciones ni la arquitectura, y ofrece los siguientes beneficios principales:

- **Ejecución duradera**: Cree agentes que persistan a pesar de los fallos y que puedan ejecutarse durante períodos prolongados, reanudando la ejecución desde donde la dejaron.
- **Control humano**: Incorporar la supervisión humana inspeccionando y modificando el estado del agente en cualquier momento.
- **Memoria integral**: Cree agentes con estado que posean tanto memoria de trabajo a corto plazo para el razonamiento continuo como memoria a largo plazo entre sesiones.
- **Depuración con LangSmith**: Obtenga una visibilidad profunda del comportamiento complejo de los agentes con herramientas de visualización que rastrean las rutas de ejecución, capturan las transiciones de estado y proporcionan métricas detalladas en tiempo de ejecución.
- **Implementación lista para producción**: implemente sistemas de agentes sofisticados con confianza gracias a una infraestructura escalable diseñada para afrontar los desafíos únicos de los flujos de trabajo con estado y de larga duración.

### Persistencia

LangGraph cuenta con una capa de persistencia integrada que guarda el estado del grafo como puntos de control. Al compilar un grafo con un sistema de puntos de control, se guarda una instantánea del estado del grafo en cada paso de la ejecución, organizada en hilos. Esto permite flujos de trabajo con intervención humana, memoria conversacional, depuración con retroceso en el tiempo y ejecución tolerante a fallos.

#### Puntos de control

El servidor de agentes gestiona automáticamente los puntos de control. Al usar el servidor de agentes, no es necesario implementar ni configurar manualmente los puntos de control. El servidor se encarga de toda la infraestructura de persistencia en segundo plano.

#### ¿Por qué usar la persistencia?

Se requiere persistencia para las siguientes funcionalidades:

**Interacción humana**: Los puntos de control facilitan los flujos de trabajo con intervención humana al permitir que los usuarios inspeccionen, interrumpan y aprueben los pasos del grafo. Estos puntos de control son necesarios para estos flujos de trabajo, ya que la persona debe poder visualizar el estado del grafo en cualquier momento, y el grafo debe poder reanudar su ejecución después de que la persona haya realizado alguna actualización de estado. Consulte la sección Interrupciones para ver ejemplos.

**Memoria**: Los puntos de control permiten mantener la memoria entre interacciones. En el caso de interacciones humanas repetidas (como conversaciones), los mensajes posteriores pueden enviarse a ese hilo, que conservará la memoria de los anteriores. Consulta "Añadir memoria" para obtener información sobre cómo añadir y gestionar la memoria de las conversaciones mediante puntos de control.

**Viaje en el tiempo**: Los puntos de control permiten "viajar en el tiempo", lo que posibilita que los usuarios reproduzcan ejecuciones previas del grafo para revisar o depurar pasos específicos del mismo. Además, los puntos de control permiten bifurcar el estado del grafo en puntos de control arbitrarios para explorar trayectorias alternativas.

**Tolerancia a fallos**: El sistema de puntos de control proporciona tolerancia a fallos y recuperación de errores: si uno o más nodos fallan en un superpaso determinado, puede reiniciar el gráfico desde el último paso exitoso.

**Escrituras pendientes**: Cuando un nodo del grafo falla durante su ejecución en un superpaso determinado, LangGraph almacena las escrituras de puntos de control pendientes de cualquier otro nodo que haya finalizado correctamente en ese superpaso. Al reanudar la ejecución del grafo desde ese superpaso, no se vuelven a ejecutar los nodos que se ejecutaron correctamente.

#### Conceptos básicos

##### Hilos

Un hilo es un identificador único asignado a cada punto de control guardado por un gestor de puntos de control. Contiene el estado acumulado de una secuencia de ejecuciones. Cuando se ejecuta una ejecución, el estado del grafo subyacente del asistente se guarda en el hilo.

Al invocar un gráfico con un punto de control, debe especificar un `thread_id` como parte de la porción `configurable` de la configuración:

```python
{"configurable": {"thread_id": "1"}}
```

Se puede recuperar el estado actual e histórico de un hilo. Para conservar el estado, es necesario crear un hilo antes de ejecutar una ejecución. La API de LangSmith proporciona varios puntos de acceso para crear y gestionar hilos y su estado. Consulte la referencia de la API para obtener más detalles.

El gestor de puntos de control utiliza `thread_id` como clave principal para almacenar y recuperar puntos de control. Sin ella, el gestor de puntos de control no puede guardar el estado ni reanudar la ejecución después de una interrupción, ya que el gestor de puntos de control utiliza `thread_id` para cargar el estado guardado.

##### Puntos de control

El estado de un hilo en un momento determinado se denomina punto de control. Un punto de control es una instantánea del estado del grafo que se guarda en cada superpaso y se representa mediante un `StateSnapshot` objeto (consulte los campos `StateSnapshot` para obtener la referencia completa del campo).

##### Superpasos

LangGraph creó un punto de control en cada límite de superpaso. Un superpaso es un único "tick" del grafo donde se ejecutan todos los nodos programados para ese paso (posiblemente en paralelo). Para un grafo secuencial como este `START -> A -> B -> END`, existen superpasos separados para la entrada, el nodo A y el nodo B, lo que produce un punto de control después de cada uno. Comprender los límites de los superpasos es importante para viajar en el tiempo, ya que solo se puede reanudar la ejecución desde un punto de control (es decir, un límite de superpaso).

Los puntos de control se conservan y pueden utilizarse para restaurar el estado de un hilo en un momento posterior.

Veamos qué puntos de control se guardan cuando se invoca un gráfico simple de la siguiente manera:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar":[]}, config)
```

Después de ejecutar el gráfico, esperamos ver exactamente 4 puntos de control:

1. Punto de control vacío con `START` como siguiente nodo a ejecutar
2. Punto de control con la entrada del usuario `{'foo': '', 'bar': []}` y `node_a` como el siguiente nodo a ejecutar
3. Punto de control con las salidas de `node_a` `{'foo': 'a', 'bar': ['a']}` y `node_b` como el siguiente nodo a ejecutar
4. Punto de control con las salidas de `node_b` `{'foo': 'b', 'bar': ['a', 'b']}` y sin nodos siguientes para ejecutar

Tenga en cuenta que los valores de `bar` del canal contienen salidas de ambos nodos, ya que tenemos un reductor para el canal `bar`.

#### Espacio de nombres de Checkpoint

Cada punto de control tiene un campo `checkpoint_ns` (espacio de nombres del punto de control) que identifica a qué gráfico o subgráfico pertenece:

- `""` (cadena vacía): El punto de control pertenece al gráfico padre (raíz).
- `"node_name:uuid"`: El punto de control pertenece a un subgrafo invocado como el nodo dado. Para subgrafos anidados, los espacios de nombres se unen con `|` separadores (por ejemplo, `"outer_node:uuid|inner_node:uuid"`).

Puedes acceder al espacio de nombres de punto de control desde dentro de un nodo a través de la configuración:

```python
from langchain_core.runnables import RunnableConfig

def my_node(state: State, config: RunnableConfig):
    checkpoint_ns = config["configurable"]["checkpoint_ns"]
    # "" for the parent graph, "node_name:uuid" for a subgraph
```

Consulte la sección Subgrafos para obtener más detalles sobre cómo trabajar con el estado y los puntos de control de los subgrafos.
​
## Obtener y actualizar el estado

### Obtener estado

Al interactuar con el estado del gráfico guardado, debe especificar un identificador de hilo. Puede ver el estado más reciente del gráfico llamando a `graph.get_state(config)`. Esto devolverá un `StateSnapshot` objeto que corresponde al último punto de control asociado con el ID de hilo proporcionado en la configuración o un punto de control asociado con un ID de punto de control para el hilo, si se proporciona.

```python
# get the latest state snapshot
config = {"configurable": {"thread_id": "1"}}
graph.get_state(config)

# get a state snapshot for a specific checkpoint_id
config = {"configurable": {"thread_id": "1", "checkpoint_id": "1ef663ba-28fe-6528-8002-5a559208592c"}}
graph.get_state(config)
```

En nuestro ejemplo, el resultado `get_state` se verá así:

```python
StateSnapshot(
    values={'foo': 'b', 'bar': ['a', 'b']},
    next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
    metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
    created_at='2024-08-29T19:19:38.821749+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}}, tasks=()
)
```

### Campos de StateSnapshot

| Campo | Tipo | Descripción |
| --- | --- | --- |
| values | dict | Valores del canal de estado en este punto de control. |
| next | tuple[str, ...] | Nombres de los nodos que se ejecutarán a continuación. Si está vacío, `()` significa que el gráfico está completo. |
| config | dict | Contiene `thread_id`, `checkpoint_ns`, y `checkpoint_id`. |
| metadata | dict | Metadatos de ejecución. Contiene `source` (`input`, `loop`, o `update`), `writes` (salidas de nodos) y `step` (contador de superpasos). |
| created_at | str | Marca de tiempo ISO 8601 que indica cuándo se creó este punto de control. |
| parent_config | dict \| None | Configuración del punto de control anterior. `None` para el primer punto de control. |
| tasks | tuple[PregelTask, ...] | Tareas a ejecutar en este paso. Cada tarea tiene `id`, `name`, `error`, `interrupts`, y opcionalmente `state` (instantánea del subgrafo, cuando se utiliza `subgraphs=True`). |

### Obtén la historia del estado

Puedes obtener el historial completo de la ejecución del grafo para un hilo determinado llamando a `graph.get_state_history(config)`. Esto devolverá una lista de `StateSnapshot` objetos asociados con el ID del hilo proporcionado en la configuración. Es importante destacar que los puntos de control se ordenarán cronológicamente, siendo el más reciente `StateSnapshot` el primero de la lista.

```python
config = {"configurable": {"thread_id": "1"}}
list(graph.get_state_history(config))
```

En nuestro ejemplo, el resultado `get_state_history` se verá así:

```python
[
    StateSnapshot(
        values={'foo': 'b', 'bar': ['a', 'b']},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28fe-6528-8002-5a559208592c'}},
        metadata={'source': 'loop', 'writes': {'node_b': {'foo': 'b', 'bar': ['b']}}, 'step': 2},
        created_at='2024-08-29T19:19:38.821749+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        tasks=(),
    ),
    StateSnapshot(
        values={'foo': 'a', 'bar': ['a']},
        next=('node_b',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f9-6ec4-8001-31981c2c39f8'}},
        metadata={'source': 'loop', 'writes': {'node_a': {'foo': 'a', 'bar': ['a']}}, 'step': 1},
        created_at='2024-08-29T19:19:38.819946+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        tasks=(PregelTask(id='6fb7314f-f114-5413-a1f3-d37dfe98ff44', name='node_b', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'foo': '', 'bar': []},
        next=('node_a',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f4-6b4a-8000-ca575a13d36a'}},
        metadata={'source': 'loop', 'writes': None, 'step': 0},
        created_at='2024-08-29T19:19:38.817813+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        tasks=(PregelTask(id='f1b14528-5ee5-579c-949b-23ef9bfbed58', name='node_a', error=None, interrupts=()),),
    ),
    StateSnapshot(
        values={'bar': []},
        next=('__start__',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1ef663ba-28f0-6c66-bfff-6723431e8481'}},
        metadata={'source': 'input', 'writes': {'foo': ''}, 'step': -1},
        created_at='2024-08-29T19:19:38.816205+00:00',
        parent_config=None,
        tasks=(PregelTask(id='6d27aa2e-d72b-5504-a36f-8620e54a76dd', name='__start__', error=None, interrupts=()),),
    )
]
```

#### Encontrar un punto de control específico

Puedes filtrar el historial de estado para encontrar puntos de control que coincidan con criterios específicos:

```python
history = list(graph.get_state_history(config))

# Find the checkpoint before a specific node executed
before_node_b = next(s for s in history if s.next == ("node_b",))

# Find a checkpoint by step number
step_2 = next(s for s in history if s.metadata["step"] == 2)

# Find checkpoints created by update_state
forks = [s for s in history if s.metadata["source"] == "update"]

# Find the checkpoint where an interrupt occurred
interrupted = next(
    s for s in history
    if s.tasks and any(t.interrupts for t in s.tasks)
)
```

#### Reproducción

La función de reproducción vuelve a ejecutar los pasos desde un punto de control anterior. Invoque el gráfico con una instrucción previa `checkpoint_id` para volver a ejecutar los nodos posteriores a dicho punto de control. Los nodos anteriores al punto de control se omiten (sus resultados ya están guardados). Los nodos posteriores al punto de control se vuelven a ejecutar, incluidas las llamadas a LLM, las solicitudes a la API o las interrupciones, que siempre se vuelven a activar durante la reproducción.

Consulte la sección Viajes en el tiempo para obtener información detallada y ejemplos de código sobre cómo reproducir ejecuciones pasadas.

#### Estado de actualización

Puedes editar el estado del gráfico usando `update_state`. Esto crea un nuevo punto de control con los valores actualizados; no modifica el punto de control original. La actualización se trata igual que una actualización de nodo: los valores se pasan a través de las funciones reductoras cuando se definen, por lo que los canales con reductores acumulan valores en lugar de sobrescribirlos.

Opcionalmente, puede especificar `as_node` de qué nodo se considera que proviene la actualización, lo que afecta al nodo que se ejecuta a continuación. Consulte Viaje en el tiempo `as_node` para obtener más detalles.

## Almacenamiento de memoria

### Modelo de estado compartido

Un esquema de estado especifica un conjunto de claves que se rellenan a medida que se ejecuta un grafo. Como se mencionó anteriormente, el estado puede escribirse mediante un puntero de control en un hilo en cada paso del grafo, lo que permite la persistencia del estado.

¿Qué ocurre si queremos conservar cierta información entre diferentes hilos de conversación? Consideremos el caso de un chatbot en el que queremos conservar información específica sobre el usuario en todas las conversaciones (por ejemplo, hilos) con ese usuario.

Con los puntos de control únicamente, no podemos compartir información entre hilos. Esto justifica la necesidad de la `Store` interfaz. A modo de ejemplo, podemos definir un objeto `InMemoryStore` para almacenar información sobre un usuario en diferentes hilos. Simplemente compilamos nuestro grafo con un punto de control, como antes, y le pasamos el objeto.

La API de LangGraph gestiona automáticamente el almacenamiento. Al usar la API de LangGraph, no es necesario implementar ni configurar el almacenamiento manualmente. La API se encarga de toda la infraestructura de almacenamiento en segundo plano.

`InMemoryStore` es adecuado para desarrollo y pruebas. Para producción, utilice un almacén persistente como `PostgresStore` o `RedisStore`. Todas las implementaciones extienden `BaseStore`, que es la anotación de tipo que se debe usar en las firmas de las funciones de nodo.

### Uso básico

Primero, vamos a mostrar esto de forma aislada, sin usar LangGraph.

```python
from langgraph.store.memory import InMemoryStore
store = InMemoryStore()
```

Las memorias se organizan mediante un espacio de nombres tuple, que en este ejemplo específico será `(<user_id>, "memories")`. El espacio de nombres puede tener cualquier longitud y representar cualquier cosa; no tiene por qué ser específico del usuario.

```python
user_id = "1"
namespace_for_memory = (user_id, "memories")
```

Usamos este `store.put` método para guardar recuerdos en nuestro espacio de nombres en el almacenamiento. Al hacerlo, especificamos el espacio de nombres, como se definió anteriormente, y un par clave-valor para el recuerdo: la clave es simplemente un identificador único para el recuerdo (`memory_id`) y el valor (un diccionario) es el recuerdo en sí.

```python
memory_id = str(uuid.uuid4())
memory = {"food_preference" : "I like pizza"}
store.put(namespace_for_memory, memory_id, memory)
```

Podemos leer los recuerdos en nuestro espacio de nombres usando el `store.search` método, que devolverá todos los recuerdos de un usuario dado como una lista. El recuerdo más reciente es el último de la lista.

```python
memories = store.search(namespace_for_memory)
memories[-1].dict()
```

```python
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

Cada tipo de memoria es una clase de Python (`Item`) con ciertos atributos. Podemos acceder a ella como un diccionario convirtiéndola `.dict` como se indicó anteriormente.

Sus atributos son:

- `value`: El valor (que a su vez es un diccionario) de esta memoria
- `key`: Una clave única para esta memoria en este espacio de nombres
- `namespace`: Una tupla de cadenas, el espacio de nombres de este tipo de memoria.
- `created_at`: Marca de tiempo de cuándo se creó este recuerdo
- `updated_at`: Marca de tiempo de cuándo se actualizó esta memoria

Si bien el tipo es `tuple[str, ...]`, puede serializarse como una lista cuando se convierte a JSON (por ejemplo, `['1', 'memories']`).

### Búsqueda semántica

Además de la simple recuperación, el almacén también admite la búsqueda semántica, lo que permite encontrar recuerdos en función del significado en lugar de coincidencias exactas. Para habilitar esta función, configure el almacén con un modelo de incrustación:

```python
from langchain.embeddings import init_embeddings

store = InMemoryStore(
    index={
        "embed": init_embeddings("openai:text-embedding-3-small"),  # Embedding provider
        "dims": 1536,                              # Embedding dimensions
        "fields": ["food_preference", "$"]              # Fields to embed
    }
)
```

Ahora, al realizar búsquedas, puedes utilizar consultas en lenguaje natural para encontrar recuerdos relevantes:

```python
# Find memories about food preferences
# (This can be done after putting memories into the store)
memories = store.search(
    namespace_for_memory,
    query="What does the user like to eat?",
    limit=3  # Return top 3 matches
)
```

Puedes controlar qué partes de tus recuerdos se incrustan configurando el `fields` parámetro o especificándolo `index` al almacenar los recuerdos:

```python
# Store with specific fields to embed
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {
        "food_preference": "I love Italian cuisine",
        "context": "Discussing dinner plans"
    },
    index=["food_preference"]  # Only embed "food_preferences" field
)

# Store without embedding (still retrievable, but not searchable)
store.put(
    namespace_for_memory,
    str(uuid.uuid4()),
    {"system_info": "Last updated: 2024-01-01"},
    index=False
)
```

## Utilizando en LangGraph

Una vez hecho esto, utilizamos el almacén en LangGraph. El almacén funciona conjuntamente con el puntero de control: el puntero de control guarda el estado en los hilos, como se explicó anteriormente, y el almacén nos permite almacenar información arbitraria para acceder a ella desde diferentes hilos. Compilamos el grafo con el puntero de control y el almacén de la siguiente manera.

```python
from dataclasses import dataclass
from langgraph.checkpoint.memory import InMemorySaver

@dataclass
class Context:
    user_id: str

# We need this because we want to enable threads (conversations)
checkpointer = InMemorySaver()

# ... Define the graph ...

# Compile the graph with the checkpointer and store
builder = StateGraph(MessagesState, context_schema=Context)
# ... add nodes and edges ...
graph = builder.compile(checkpointer=checkpointer, store=store)
```

Invocamos el gráfico con un `thread_id`, como antes, y también con un `user_id`, que utilizaremos para asignar nombres a nuestros recuerdos para este usuario en particular, como mostramos anteriormente.

```python
# Invoke the graph
config = {"configurable": {"thread_id": "1"}}

# First let's just say hi to the AI
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),
):
    print(update)
```

Puedes acceder al almacenamiento y a la memoria `user_id` en cualquier nodo usando el `Runtime` objeto. `Runtime` LangGraph lo inyecta automáticamente cuando lo agregas como parámetro a la función de tu nodo. Así es como podrías usarlo para guardar memorias:

```python
from langgraph.runtime import Runtime
from dataclasses import dataclass

@dataclass
class Context:
    user_id: str

async def update_memory(state: MessagesState, runtime: Runtime[Context]):

    # Get the user id from the runtime context
    user_id = runtime.context.user_id

    # Namespace the memory
    namespace = (user_id, "memories")

    # ... Analyze conversation and create a new memory

    # Create a new memory ID
    memory_id = str(uuid.uuid4())

    # We create a new memory
    await runtime.store.aput(namespace, memory_id, {"memory": memory})
```

Como mostramos anteriormente, también podemos acceder al almacenamiento en cualquier nodo y usar el `store.search` método para obtener recuerdos. Recordemos que los recuerdos se devuelven como una lista de objetos que se pueden convertir en un diccionario.

```python
memories[-1].dict()
```

```python
{'value': {'food_preference': 'I like pizza'},
 'key': '07e0caf4-1631-47b7-b15f-65515d4c1843',
 'namespace': ['1', 'memories'],
 'created_at': '2024-10-02T17:22:31.590602+00:00',
 'updated_at': '2024-10-02T17:22:31.590605+00:00'}
```

Podemos acceder a los recuerdos y utilizarlos en nuestra llamada al modelo.

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    # Get the user id from the runtime context
    user_id = runtime.context.user_id

    # Namespace the memory
    namespace = (user_id, "memories")

    # Search based on the most recent message
    memories = await runtime.store.asearch(
        namespace,
        query=state["messages"][-1].content,
        limit=3
    )
    info = "\n".join([d.value["memory"] for d in memories])

    # ... Use memories in the model call
```

Si creamos un nuevo hilo, aún podemos acceder a las mismas memorias siempre y cuando `user_id` sea el mismo.

```python
# Invoke the graph on a new thread
config = {"configurable": {"thread_id": "2"}}

# Let's say hi again
for update in graph.stream(
    {"messages": [{"role": "user", "content": "hi, tell me about my memories"}]},
    config,
    stream_mode="updates",
    context=Context(user_id="1"),
):
    print(update)
```

Cuando usamos LangSmith, ya sea localmente (por ejemplo, en Studio) o alojado con LangSmith, el almacén base está disponible por defecto y no es necesario especificarlo durante la compilación del gráfico. Sin embargo, para habilitar la búsqueda semántica, es necesario configurar los ajustes de indexación en el archivo `langgraph.json`. Por ejemplo:

```json
{
    ...
    "store": {
        "index": {
            "embed": "openai:text-embeddings-3-small",
            "dims": 1536,
            "fields": ["$"]
        }
    }
}
```

Consulte la guía de implementación para obtener más detalles y opciones de configuración.

## Bibliotecas de puntos de control

Internamente, el sistema de puntos de control se basa en objetos de puntero de control que cumplen con `BaseCheckpointSaver` una interfaz. LangGraph proporciona varias implementaciones de punteros de control, todas ellas implementadas mediante bibliotecas independientes e instalables.

Consulte las integraciones de Checkpointer para ver los proveedores disponibles.

- `langgraph-checkpoint`: La interfaz base para los guardadores de puntos de control (`BaseCheckpointSaver`) y la interfaz de serialización/deserialización (`SerializerProtocol`). Incluye una implementación de puntos de control en memoria (`InMemorySaver`) para experimentación. LangGraph viene `langgraph-checkpoint` incluido.
- `langgraph-checkpoint-sqlite`: Implementación del puntero de control LangGraph que utiliza una base de datos SQLite (`SqliteSaver` / `AsyncSqliteSaver`). Ideal para experimentación y flujos de trabajo locales. Requiere instalación por separado.
- `langgraph-checkpoint-postgres`: Un sistema avanzado de gestión de puntos de control que utiliza la base de datos Postgres (`PostgresSaver` / `AsyncPostgresSaver`), utilizado en LangSmith. Ideal para su uso en producción. Debe instalarse por separado.
- `langgraph-checkpoint-cosmosdb`: Implementación de un puntero de control LangGraph que utiliza Azure Cosmos DB (`CosmosDBSaver` / `AsyncCosmosDBSaver`). Ideal para su uso en producción con Azure. Admite operaciones síncronas y asíncronas. Requiere instalación por separado.

## Interfaz de puntero de control

Cada punto de control se ajusta a `BaseCheckpointSaver` la interfaz e implementa los siguientes métodos:

- `.put`: Almacenar un punto de control con su configuración y metadatos.
- `.put_writes`: Almacenar escrituras intermedias vinculadas a un punto de control (es decir, escrituras pendientes).
- `.get_tuple`: Obtener una tupla de punto de control usando para una configuración dada (`thread_id` y `checkpoint_id`). Esto se usa para rellenar `StateSnapshot` en `graph.get_state()`.
- `.list`: Enumerar los puntos de control que coinciden con una configuración y criterios de filtro determinados. Esto se utiliza para completar el historial de estado en `graph.get_state_history()`.

Si se utiliza el punto de control con ejecución de grafo asíncrona (es decir, ejecutando el grafo a través de `.ainvoke`, `.astream`, `.abatch`), se utilizarán versiones asíncronas de los métodos anteriores (`.aput`, `.aput_writes`, `.aget_tuple`, `.alist`).

Para ejecutar su gráfico de forma asíncrona, puede usar `InMemorySaver`, o versiones asíncronas de los puntos de control de Sqlite/Postgres — `AsyncSqliteSaver` / `AsyncPostgresSaver` checkpointers.

## Serializador

Cuando los puntos de control guardan el estado del gráfico, necesitan serializar los valores de los canales en dicho estado. Esto se realiza mediante objetos serializadores.

`langgraph_checkpoint` define un protocolo para implementar serializadores y proporciona una implementación predeterminada (`JsonPlusSerializer`) que maneja una amplia variedad de tipos, incluyendo primitivas LangChain y LangGraph, fechas y horas, enumeraciones y más.

### Serialización con pickle

El serializador predeterminado `JsonPlusSerializer` utiliza `ormsgpack` y JSON internamente, lo cual no es adecuado para todos los tipos de objetos.

Si desea recurrir a pickle para objetos que actualmente no son compatibles con nuestro codificador msgpack (como dataframes de Pandas), puede usar el `pickle_fallback` argumento de `JsonPlusSerializer`:

```python
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

# ... Define the graph ...
graph.compile(
    checkpointer=InMemorySaver(serde=JsonPlusSerializer(pickle_fallback=True))
)
```

## Cifrado

Los puntos de control pueden cifrar opcionalmente todo el estado persistente. Para habilitar esto, pase una instancia de `EncryptedSerializer` al `serde` argumento de cualquier `BaseCheckpointSaver` implementación. La forma más sencilla de crear un serializador cifrado es mediante `from_pycryptodome_aes`, que lee la clave AES de la `LANGGRAPH_AES_KEY` variable de entorno (o acepta un `key` argumento):

```python
import sqlite3

from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.sqlite import SqliteSaver

serde = EncryptedSerializer.from_pycryptodome_aes()  # reads LANGGRAPH_AES_KEY
checkpointer = SqliteSaver(sqlite3.connect("checkpoint.db"), serde=serde)
```

```python
from langgraph.checkpoint.serde.encrypted import EncryptedSerializer
from langgraph.checkpoint.postgres import PostgresSaver

serde = EncryptedSerializer.from_pycryptodome_aes()
checkpointer = PostgresSaver.from_conn_string("postgresql://...", serde=serde)
checkpointer.setup()
```

Cuando se ejecuta en LangSmith, el cifrado se habilita automáticamente siempre que `LANGGRAPH_AES_KEY` esté presente, por lo que solo necesita proporcionar la variable de entorno. Se pueden utilizar otros esquemas de cifrado implementándolos `CipherProtocol` y proporcionándolos a `EncryptedSerializer`.

## Transmisión

LangGraph implementa un sistema de transmisión para mostrar actualizaciones en tiempo real. La transmisión es fundamental para mejorar la capacidad de respuesta de las aplicaciones basadas en LLM. Al mostrar la salida progresivamente, incluso antes de que la respuesta completa esté lista, la transmisión mejora significativamente la experiencia del usuario (UX), especialmente al lidiar con la latencia de los LLM.

### Uso básico

Los gráficos de LangGraph exponen los métodos `stream` (síncrono) y `astream` (asíncrono) para generar salidas en flujo continuo como iteradores. Pase uno o más modos de flujo para controlar qué datos recibirá.

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

**Producción**

```text
Status: thinking of a joke...
Node generate_joke updated: {'joke': 'Why did the ice cream go to school? To get a sundae education!'}
```

### Ejemplo completo

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.config import get_stream_writer

class State(TypedDict):
    topic: str
    joke: str


def generate_joke(state: State):
    writer = get_stream_writer()
    writer({"status": "thinking of a joke..."})
    return {"joke": f"Why did the {state['topic']} go to school? To get a sundae education!"}

graph = (
    StateGraph(State)
    .add_node(generate_joke)
    .add_edge(START, "generate_joke")
    .add_edge("generate_joke", END)
    .compile()
)

for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["updates", "custom"],
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node {node_name} updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Status: {chunk['data']['status']}")
```

**Producción**

```text
Status: thinking of a joke...
Node generate_joke updated: {'joke': 'Why did the ice cream go to school? To get a sundae education!'}
```

### Formato de salida de flujo (v2)

Requiere LangGraph >= 1.1. Todos los ejemplos de esta página utilizan `version="v2"`.

Pase `version="v2"` a `stream()` o a `astream()` para obtener un formato de salida unificado. Cada fragmento es un `StreamPart` diccionario con una forma consistente, independientemente del modo de flujo, el número de modos o la configuración del subgrafo:

```python
{
    "type": "values" | "updates" | "messages" | "custom" | "checkpoints" | "tasks" | "debug",
    "ns": (),           # namespace tuple, populated for subgraph events
    "data": ...,        # the actual payload (type varies by stream mode)
}
```

Cada modo de flujo tiene un `TypedDict` contenedor correspondiente `ValuesStreamPart`, `UpdatesStreamPart`, `MessagesStreamPart`, `CustomStreamPart`, `CheckpointStreamPart`, `TasksStreamPart`, `DebugStreamPart`. Puede importar estos tipos desde `langgraph.types`. El tipo de unión `StreamPart` es una unión disjunta en `part["type"]`, lo que permite una restricción de tipo completa en editores y verificadores de tipos.

Con la versión 1 (predeterminada), el formato de salida cambia según las opciones de transmisión (el modo único devuelve datos sin procesar, los modos múltiples devuelven `(mode, data)` tuplas, los subgrafos devuelven `(namespace, data)` tuplas). Con la versión 2, el formato siempre es el mismo:

```python
for chunk in graph.stream(inputs, stream_mode="updates", version="v2"):
    print(chunk["type"])  # "updates"
    print(chunk["ns"])    # ()
    print(chunk["data"])  # {"node_name": {"key": "value"}}
```

El formato v2 también permite la restricción de tipos, lo que significa que puede filtrar fragmentos `chunk["type"]` y obtener el tipo de carga útil correcto. Cada rama se restringe `part["data"]` al tipo específico para ese modo:

```python
for part in graph.stream(
    {"topic": "ice cream"},
    stream_mode=["values", "updates", "messages", "custom"],
    version="v2",
):
    if part["type"] == "values":
        # ValuesStreamPart — full state snapshot after each step
        print(f"State: topic={part['data']['topic']}")
    elif part["type"] == "updates":
        # UpdatesStreamPart — only the changed keys from each node
        for node_name, state in part["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif part["type"] == "messages":
        # MessagesStreamPart — (message_chunk, metadata) from LLM calls
        msg, metadata = part["data"]
        print(msg.content, end="", flush=True)
    elif part["type"] == "custom":
        # CustomStreamPart — arbitrary data from get_stream_writer()
        print(f"Progress: {part['data']['progress']}%")
```

### Modos de transmisión

Pase uno o más de los siguientes modos de flujo como una lista a los métodos `stream` o `astream`.

| Modo | Tipo | Descripción |
| --- | --- | --- |
| valores | `ValuesStreamPart` | Estado completo después de cada paso. |
| actualizaciones | `UpdatesStreamPart` | El estado se actualiza después de cada paso. Las actualizaciones múltiples en el mismo paso se transmiten por separado. |
| mensajes | `MessagesStreamPart` | Tuplas de 2 elementos (token LLM, metadatos) de llamadas LLM. |
| costumbre | `CustomStreamPart` | Datos personalizados emitidos desde nodos a través de `get_stream_writer`. |
| puntos de control | `CheckpointStreamPart` | Eventos de punto de control (mismo formato que `get_state()`). Requiere un gestor de puntos de control. |
| tareas | `TasksStreamPart` | Eventos de inicio y finalización de tareas con resultados y errores. Requiere un gestor de puntos de control. |
| depurar | `DebugStreamPart` | Toda la información disponible, combinada `checkpoints` con `tasks`, metadatos adicionales. |

## Estado del gráfico

Utilice los modos de transmisión `updates` para `values` transmitir el estado del gráfico mientras se ejecuta.

- `updates` transmite las actualizaciones del estado después de cada paso del gráfico.
- `values` transmite el valor completo del estado después de cada paso del gráfico.

```python
from typing import TypedDict
from langgraph.graph import StateGraph, START, END


class State(TypedDict):
  topic: str
  joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


def generate_joke(state: State):
    return {"joke": f"This is a joke about {state['topic']}"}

graph = (
  StateGraph(State)
  .add_node(refine_topic)
  .add_node(generate_joke)
  .add_edge(START, "refine_topic")
  .add_edge("refine_topic", "generate_joke")
  .add_edge("generate_joke", END)
  .compile()
)
```

Utilice esta función para transmitir únicamente las actualizaciones de estado que devuelven los nodos después de cada paso. Las salidas transmitidas incluyen el nombre del nodo, así como la actualización.

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="updates",
    version="v2",
):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
```

**Producción**

```text
Node `refine_topic` updated: {'topic': 'ice cream and cats'}
Node `generate_joke` updated: {'joke': 'This is a joke about ice cream and cats'}
```

tokens LLM
## tokens LLM

Utilice el modo `messages` de transmisión para transmitir las salidas del Modelo de Lenguaje Grande (LLM, por sus siglas en inglés) token por token desde cualquier parte de su grafo, incluidos nodos, herramientas, subgrafos o tareas.

La salida transmitida desde el modo `messages` es una tupla `(message_chunk, metadata)` donde:

- `message_chunk`: el token o segmento de mensaje del LLM.
- `metadata`: un diccionario que contiene detalles sobre el nodo del gráfico y la invocación de LLM.

Si su LLM no está disponible como integración de LangChain, puede transmitir sus resultados utilizando el modo `custom` correspondiente. Consulte la sección "Uso con cualquier LLM" para obtener más detalles.

Se requiere configuración manual para la programación asíncrona en Python < 3.11. Al usar Python < 3.11 con código asíncrono, debe pasar explícitamente un parámetro `RunnableConfig` para `ainvoke()` y habilitar la transmisión correcta. Consulte la sección "Programación asíncrona con Python < 3.11" para obtener más detalles o actualice a Python 3.11 o superior.

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START

@dataclass
class MyState:
    topic: str
    joke: str = ""


model = init_chat_model(model="gpt-4.1-mini")

def call_model(state: MyState):
    """Call the LLM to generate a joke about a topic"""
    # Note that message events are emitted even when the LLM is run using .invoke rather than .stream
    model_response = model.invoke(
        [
            {"role": "user", "content": f"Generate a joke about {state.topic}"}
        ]
    )
    return {"joke": model_response.content}

graph = (
    StateGraph(MyState)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)
```

Utilice `stream_mode="messages"` para transmitir tokens LLM con metadatos. Use `version="v2"` para un formato `StreamPart` unificado.

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
```

Filtrar por invocación de LLM
Puedes asociarte tagsa las invocaciones de LLM para filtrar los tokens transmitidos según la invocación de LLM.

```python
from langchain.chat_models import init_chat_model

# model_1 is tagged with "joke"
model_1 = init_chat_model(model="gpt-4.1-mini", tags=['joke'])
# model_2 is tagged with "poem"
model_2 = init_chat_model(model="gpt-4.1-mini", tags=['poem'])

graph = ... # define a graph that uses these LLMs
```

# The stream_mode is set to "messages" to stream LLM tokens
# The metadata contains information about the LLM invocation, including the tags

```python
async for chunk in graph.astream(
    {"topic": "cats"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Filter the streamed tokens by the tags field in the metadata to only include
        # the tokens from the LLM invocation with the "joke" tag
        if metadata["tags"] == ["joke"]:
            print(msg.content, end="|", flush=True)
```

Ejemplo ampliado: filtrado por etiquetas

​
Omitir mensajes del flujo
Utilice la nostreametiqueta para excluir completamente la salida de LLM del flujo. Las invocaciones etiquetadas con nostreamaún se ejecutan y producen salida; sus tokens simplemente no se emiten en messagesmodo.
Esto resulta útil cuando:
Necesitas la salida de LLM para procesamiento interno (por ejemplo, salida estructurada) pero no quieres transmitirla al cliente.
Transmites el mismo contenido a través de un canal diferente (por ejemplo, mensajes de interfaz de usuario personalizados) y quieres evitar la duplicación de la salida en la messagestransmisión.

```python
from typing import Any, TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import START, StateGraph

stream_model = ChatAnthropic(model_name="claude-3-haiku-20240307")
internal_model = ChatAnthropic(model_name="claude-3-haiku-20240307").with_config(
    {"tags": ["nostream"]}
)


class State(TypedDict):
    topic: str
    answer: str
    notes: str


def answer(state: State) -> dict[str, Any]:
    r = stream_model.invoke(
        [{"role": "user", "content": f"Reply briefly about {state['topic']}"}]
    )
    return {"answer": r.content}


def internal_notes(state: State) -> dict[str, Any]:
    # Tokens from this model are omitted from stream_mode="messages" because of nostream
    r = internal_model.invoke(
        [{"role": "user", "content": f"Private notes on {state['topic']}"}]
    )
    return {"notes": r.content}


graph = (
    StateGraph(State)
    .add_node("write_answer", answer)
    .add_node("internal_notes", internal_notes)
    .add_edge(START, "write_answer")
    .add_edge("write_answer", "internal_notes")
    .compile()
)

initial_state: State = {"topic": "AI", "answer": "", "notes": ""}
stream = graph.stream(initial_state, stream_mode="messages")
```

Filtrar por nodo
Para transmitir tokens solo desde nodos específicos, utilice `stream_mode="messages"` y filtre las salidas por el campo `langgraph_node` en los metadatos transmitidos.

Utilice `stream_mode="messages"` para transmitir tokens LLM con metadatos. Use `version="v2"` para un formato `StreamPart` unificado.

```python
for chunk in graph.stream(
    inputs,
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Filter the streamed tokens by the langgraph_node field in the metadata
        # to only include the tokens from the specified node
        if msg.content and metadata["langgraph_node"] == "some_node_name":
            ...
```

Ejemplo ampliado: transmisión de tokens LLM desde nodos específicos

```python
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-4.1-mini")


class State(TypedDict):
      topic: str
      joke: str
      poem: str


def write_joke(state: State):
      topic = state["topic"]
      joke_response = model.invoke(
            [{"role": "user", "content": f"Write a joke about {topic}"}]
      )
      return {"joke": joke_response.content}


def write_poem(state: State):
      topic = state["topic"]
      poem_response = model.invoke(
            [{"role": "user", "content": f"Write a short poem about {topic}"}]
      )
      return {"poem": poem_response.content}


graph = (
      StateGraph(State)
      .add_node(write_joke)
      .add_node(write_poem)
      # write both the joke and the poem concurrently
      .add_edge(START, "write_joke")
      .add_edge(START, "write_poem")
      .compile()
)
```

# The "messages" stream mode streams LLM tokens with metadata
# Use version="v2" for a unified StreamPart format

```python
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        msg, metadata = chunk["data"]
        # Filter the streamed tokens by the langgraph_node field in the metadata
        # to only include the tokens from the write_poem node
        if msg.content and metadata["langgraph_node"] == "write_poem":
            print(msg.content, end="|", flush=True)
```

Datos personalizados
Para enviar datos personalizados definidos por el usuario desde dentro de un nodo o herramienta de LangGraph, siga estos pasos:
Se utiliza get_stream_writerpara acceder al escritor de flujo y emitir datos personalizados.
Establézcalo stream_mode="custom"al llamar .stream()o .astream()para obtener los datos personalizados en el flujo. Puede combinar varios modos (por ejemplo, ["updates", "custom"]), pero al menos uno debe ser "custom".
No get_stream_writerfunciona en async para Python < 3.11. En código async que se ejecuta en Python < 3.11, get_stream_writerno funcionará. En su lugar, agregue un writerparámetro a su nodo o herramienta y páselo manualmente. Consulte Async con Python < 3.11 para ver ejemplos de uso.

```python
from typing import TypedDict
from langgraph.config import get_stream_writer
from langgraph.graph import StateGraph, START

class State(TypedDict):
    query: str
    answer: str

def node(state: State):
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Emit a custom key-value pair (e.g., progress update)
    writer({"custom_key": "Generating custom data inside node"})
    return {"answer": "some data"}

graph = (
    StateGraph(State)
    .add_node(node)
    .add_edge(START, "node")
    .compile()
)

inputs = {"query": "example"}
```

# Set stream_mode="custom" to receive the custom data in the stream

```python
for chunk in graph.stream(inputs, stream_mode="custom", version="v2"):
    if chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']['custom_key']}")
​```

Salidas de subgrafos
Para incluir las salidas de los subgrafos en las salidas transmitidas, puede configurarlo subgraphs=Trueen el .stream()método del grafo principal. Esto transmitirá las salidas tanto del grafo principal como de cualquier subgrafo.
Las salidas se transmitirán como tuplas (namespace, data), donde namespacees una tupla con la ruta al nodo donde se invoca un subgrafo, por ejemplo ("parent_node:<task_id>", "child_node:<task_id>").

Con version="v2", los eventos de subgrafo utilizan el mismo StreamPartformato. El nscampo identifica la fuente:

```python
for chunk in graph.stream(
    {"foo": "foo"},
    subgraphs=True,
    stream_mode="updates",
    version="v2",
):
    print(chunk["type"])  # "updates"
    print(chunk["ns"])    # () for root, ("node_name:<task_id>",) for subgraph
    print(chunk["data"])  # {"node_name": {"key": "value"}}
```

Ejemplo ampliado: transmisión desde subgrafos

```python
from langgraph.graph import START, StateGraph
from typing import TypedDict

# Define subgraph
class SubgraphState(TypedDict):
    foo: str  # note that this key is shared with the parent graph state
    bar: str

def subgraph_node_1(state: SubgraphState):
    return {"bar": "bar"}

def subgraph_node_2(state: SubgraphState):
    return {"foo": state["foo"] + state["bar"]}

subgraph_builder = StateGraph(SubgraphState)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_node(subgraph_node_2)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph_builder.add_edge("subgraph_node_1", "subgraph_node_2")
subgraph = subgraph_builder.compile()

# Define parent graph
class ParentState(TypedDict):
    foo: str

def node_1(state: ParentState):
    return {"foo": "hi! " + state["foo"]}

builder = StateGraph(ParentState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", subgraph)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
graph = builder.compile()

for chunk in graph.stream(
    {"foo": "foo"},
    stream_mode="updates",
    # Set subgraphs=True to stream outputs from subgraphs
    subgraphs=True,
    version="v2",
):
    if chunk["type"] == "updates":
        if chunk["ns"]:
            print(f"Subgraph {chunk['ns']}: {chunk['data']}")
        else:
            print(f"Root: {chunk['data']}")
Root: {'node_1': {'foo': 'hi! foo'}}
Subgraph ('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',): {'subgraph_node_1': {'bar': 'bar'}}
Subgraph ('node_2:dfddc4ba-c3c5-6887-5012-a243b5b377c2',): {'subgraph_node_2': {'foo': 'hi! foobar'}}
Root: {'node_2': {'foo': 'hi! foobar'}}
```

Tenga en cuenta que no solo recibimos las actualizaciones de los nodos, sino también los espacios de nombres que nos indican de qué grafo (o subgrafo) estamos transmitiendo la información.
​
Puntos de control
Utilice el checkpointsmodo de transmisión para recibir eventos de punto de control a medida que se ejecuta el gráfico. Cada evento de punto de control tiene el mismo formato que la salida de get_state(). Requiere un generador de puntos de control.

```python
from langgraph.checkpoint.memory import MemorySaver

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="checkpoints",
    version="v2",
):
    if chunk["type"] == "checkpoints":
        print(chunk["data"])
```

Tareas
Utilice el tasksmodo de transmisión para recibir eventos de inicio y finalización de tareas a medida que se ejecuta el gráfico. Los eventos de tarea incluyen información sobre qué nodo se está ejecutando, sus resultados y cualquier error. Requiere un gestor de puntos de control .

```python
from langgraph.checkpoint.memory import MemorySaver

graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .add_edge("generate_joke", END)
    .compile(checkpointer=MemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

for chunk in graph.stream(
    {"topic": "ice cream"},
    config=config,
    stream_mode="tasks",
    version="v2",
):
    if chunk["type"] == "tasks":
        print(chunk["data"])
```

Depurar
Utilice el debugmodo de transmisión continua para transmitir la mayor cantidad de información posible durante la ejecución del gráfico. Las salidas transmitidas incluyen el nombre del nodo, así como su estado completo.

```python
for chunk in graph.stream(
    {"topic": "ice cream"},
    stream_mode="debug",
    version="v2",
):
    if chunk["type"] == "debug":
        print(chunk["data"])
```
Este debugmodo combina checkpointseventos taskscon metadatos adicionales. Úselo checkpointsdirectamente taskssi solo necesita un subconjunto de la información de depuración.
​
Múltiples modos a la vez
Puedes pasar una lista como stream_modeparámetro para transmitir varios modos a la vez.
Con version="v2", cada fragmento es un StreamPartdiccionario. Úselo chunk["type"]para distinguir entre modos:

```python
for chunk in graph.stream(inputs, stream_mode=["updates", "custom"], version="v2"):
    if chunk["type"] == "updates":
        for node_name, state in chunk["data"].items():
            print(f"Node `{node_name}` updated: {state}")
    elif chunk["type"] == "custom":
        print(f"Custom event: {chunk['data']}")
​```


Avanzado

Utilizar con cualquier LLM
Puedes usarlo stream_mode="custom"para transmitir datos desde cualquier API de LLM , incluso si esa API no implementa la interfaz del modelo de chat de LangChain.
Esto permite integrar clientes LLM sin procesar o servicios externos que proporcionan sus propias interfaces de transmisión, lo que hace que LangGraph sea muy flexible para configuraciones personalizadas.

```python
from langgraph.config import get_stream_writer

def call_arbitrary_model(state):
    """Example node that calls an arbitrary model and streams the output"""
    # Get the stream writer to send custom data
    writer = get_stream_writer()
    # Assume you have a streaming client that yields chunks
    # Generate LLM tokens using your custom streaming client
    for chunk in your_custom_streaming_client(state["topic"]):
        # Use the writer to send custom data to the stream
        writer({"custom_llm_chunk": chunk})
    return {"result": "completed"}

graph = (
    StateGraph(State)
    .add_node(call_arbitrary_model)
    # Add other nodes and edges as needed
    .compile()
)
# Set stream_mode="custom" to receive the custom data in the stream
for chunk in graph.stream(
    {"topic": "cats"},
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        # The chunk data will contain the custom data streamed from the llm
        print(chunk["data"])
```

Ejemplo ampliado: modelo de chat arbitrario en tiempo real

```python
import operator
import json

from typing import TypedDict
from typing_extensions import Annotated
from langgraph.graph import StateGraph, START

from openai import AsyncOpenAI

openai_client = AsyncOpenAI()
model_name = "gpt-4.1-mini"

async def stream_tokens(model_name: str, messages: list[dict]):
    response = await openai_client.chat.completions.create(
        messages=messages, model=model_name, stream=True
    )
    role = None
    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.role is not None:
            role = delta.role

        if delta.content:
            yield {"role": role, "content": delta.content}


# this is our tool
async def get_items(place: str) -> str:
    """Use this tool to list items one might find in a place you're asked about."""
    writer = get_stream_writer()
    response = ""
    async for msg_chunk in stream_tokens(
        model_name,
        [
            {
                "role": "user",
                "content": (
                    "Can you tell me what kind of items "
                    f"i might find in the following place: '{place}'. "
                    "List at least 3 such items separating them by a comma. "
                    "And include a brief description of each item."
                ),
            }
        ],
    ):
        response += msg_chunk["content"]
        writer(msg_chunk)

    return response


class State(TypedDict):
    messages: Annotated[list[dict], operator.add]

# this is the tool-calling graph node
async def call_tool(state: State):
    ai_message = state["messages"][-1]
    tool_call = ai_message["tool_calls"][-1]

    function_name = tool_call["function"]["name"]
    if function_name != "get_items":
        raise ValueError(f"Tool {function_name} not supported")

    function_arguments = tool_call["function"]["arguments"]
    arguments = json.loads(function_arguments)

    function_response = await get_items(**arguments)
    tool_message = {
        "tool_call_id": tool_call["id"],
        "role": "tool",
        "name": function_name,
        "content": function_response,
    }
    return {"messages": [tool_message]}

graph = (
    StateGraph(State)
    .add_node(call_tool)
    .add_edge(START, "call_tool")
    .compile()
)

Vamos a invocar el gráfico con una `AIMessage` llamada a una herramienta:

```python
inputs = {
    "messages": [
        {
            "content": None,
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "1",
                    "function": {
                        "arguments": '{"place":"bedroom"}',
                        "name": "get_items",
                    },
                    "type": "function",
                }
            ],
        }
    ]
}

async for chunk in graph.astream(
    inputs,
    stream_mode="custom",
    version="v2",
):
    if chunk["type"] == "custom":
        print(chunk["data"]["content"], end="|", flush=True)
```
​
Deshabilitar la transmisión para modelos de chat específicos
Si su aplicación combina modelos que admiten la transmisión de datos con otros que no, es posible que deba deshabilitar explícitamente la transmisión de datos para los modelos que no la admiten.
Se configura streaming=Falseal inicializar el modelo.
modelo_de_chat_inicial
Interfaz del modelo de chat

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "claude-sonnet-4-6",
    # Set streaming=False to disable streaming for the chat model
    streaming=False
)
```

No todas las integraciones de modelos de chat admiten este streamingparámetro. Si tu modelo no lo admite, usa disable_streaming=Trueen su lugar. Este parámetro está disponible en todos los modelos de chat a través de la clase base.
## Migrar a la versión 2

El formato de transmisión v2 (utilizado en toda esta página) proporciona un formato de salida unificado. A continuación, se presenta un resumen de las principales diferencias y cómo migrar:

| Concepto | v1 (predeterminado) | v2 (`version="v2"`) |
|----------|---------------------|----------------------|
| Modo de flujo único | Datos sin procesar (diccionario) | `StreamPart` diccionario con `type`, `ns`, `data` |
| Múltiples modos de transmisión | `(mode, data)` tuplas | Mismo `StreamPart` diccionario, filtrar por `chunk["type"]` |
| Transmisión de subgrafos | `(namespace, data)` tuplas | Mismo `StreamPart` diccionario, comprueba `chunk["ns"]` |
| Múltiples modos + subgrafos | `(namespace, mode, data)` tríos | Mismo `StreamPart` diccionario |
| `invoke()` tipo de retorno | Diccionario simple (estado) | `GraphOutput` con `.value` y `.interrupts` |
| Ubicación de interrupción (flujo) | `__interrupt__` clave en el diccionario de estado | `interrupts` campo en values partes de arroyo |
| Ubicación de interrupción (invocar) | `__interrupt__` clave en el diccionario de resultados | `.interrupts` atributo en `GraphOutput` |
| Salida de Pydantic/dataclass | Devuelve un diccionario simple | Convierte a instancia de modelo/clase de datos |

### Formato de invocación v2

Cuando se pasa `version="v2"` a `invoke()` o `ainvoke()`, devuelve un `GraphOutput` objeto con atributos `.value` y `.interrupts`:

```python
from langgraph.types import GraphOutput

result = graph.invoke(inputs, version="v2")

assert isinstance(result, GraphOutput)
result.value       # your output — dict, Pydantic model, or dataclass
result.interrupts  # tuple[Interrupt, ...], empty if none occurred
Con cualquier modo de transmisión distinto al predeterminado "values", invoke(..., stream_mode="updates", version="v2")devuelve list[StreamPart]en lugar de list[tuple].
El acceso estilo diccionario en GraphOutput( result["key"], "key" in result, result["__interrupt__"]) todavía funciona para compatibilidad con versiones anteriores, pero está obsoleto y se eliminará en una versión futura. Migre a result.valuey result.interrupts.
Esto separa el estado de los metadatos de interrupción. Con v1, las interrupciones se incrustan en el diccionario devuelto bajo __interrupt__:

config = {"configurable": {"thread_id": "thread-1"}}
result = graph.invoke(inputs, config=config, version="v2")

if result.interrupts:
    print(result.interrupts[0].value)
    graph.invoke(Command(resume=True), config=config, version="v2")
​
```


Pydantic y coerción de estado de clase de datos
Cuando el estado de su gráfico es un modelo o clase de datos de Pydantic, valuesel modo v2 convierte automáticamente la salida al tipo correcto:

```python
from pydantic import BaseModel
from typing import Annotated
import operator

class MyState(BaseModel):
    value: str
    items: Annotated[list[str], operator.add]

# With version="v2", chunk["data"] is a MyState instance
for chunk in graph.stream(
    {"value": "x", "items": []}, stream_mode="values", version="v2"
):
    print(type(chunk["data"]))  # <class 'MyState'>
​```

Asíncrono con Python < 3.11
En las versiones de Python < 3.11, las tareas asyncio no admiten el contextparámetro. Esto limita la capacidad de LangGraph para propagar automáticamente el contexto y afecta a los mecanismos de transmisión de LangGraph de dos maneras clave:
Debe pasar explícitamente RunnableConfiga las llamadas LLM asíncronas (por ejemplo, ainvoke()), ya que las devoluciones de llamada no se propagan automáticamente.
No se puede usar get_stream_writeren nodos o herramientas asíncronas; debe pasar un writerargumento directamente.
Ejemplo ampliado: llamada LLM asíncrona con configuración manual

```python
from typing import TypedDict
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model

model = init_chat_model(model="gpt-4.1-mini")

class State(TypedDict):
    topic: str
    joke: str

# Accept config as an argument in the async node function
async def call_model(state, config):
    topic = state["topic"]
    print("Generating joke...")
    # Pass config to model.ainvoke() to ensure proper context propagation
    joke_response = await model.ainvoke(
        [{"role": "user", "content": f"Write a joke about {topic}"}],
        config,
    )
    return {"joke": joke_response.content}

graph = (
    StateGraph(State)
    .add_node(call_model)
    .add_edge(START, "call_model")
    .compile()
)

# Set stream_mode="messages" to stream LLM tokens
async for chunk in graph.astream(
    {"topic": "ice cream"},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        message_chunk, metadata = chunk["data"]
        if message_chunk.content:
            print(message_chunk.content, end="|", flush=True)
Ejemplo ampliado: transmisión personalizada asíncrona con escritor de flujo

from typing import TypedDict
from langgraph.types import StreamWriter

class State(TypedDict):
      topic: str
      joke: str

# Add writer as an argument in the function signature of the async node or tool
# LangGraph will automatically pass the stream writer to the function
async def generate_joke(state: State, writer: StreamWriter):
      writer({"custom_key": "Streaming custom data while generating a joke"})
      return {"joke": f"This is a joke about {state['topic']}"}

graph = (
      StateGraph(State)
      .add_node(generate_joke)
      .add_edge(START, "generate_joke")
      .compile()
)

# Set stream_mode="custom" to receive the custom data in the stream  #
async for chunk in graph.astream(
      {"topic": "ice cream"},
      stream_mode="custom",
      version="v2",
):
      if chunk["type"] == "custom":
          print(chunk["data"])
```

### Utilizar los viajes en el tiempo

​LangGraph admite viajes en el tiempo a través de puntos de control :
Repetición : Reintentar desde un punto de control anterior.
Bifurcación : Crea una rama a partir de un punto de control anterior con un estado modificado para explorar una ruta alternativa.
Ambos métodos funcionan reanudando la ejecución desde un punto de control anterior. Los nodos anteriores al punto de control no se vuelven a ejecutar (los resultados ya están guardados). Los nodos posteriores al punto de control se vuelven a ejecutar, incluyendo las llamadas a LLM, las solicitudes a la API y las interrupciones (que pueden producir resultados diferentes).
​
Repetición
Invoca el gráfico con la configuración de un punto de control anterior para reproducir desde ese punto.
La función de reproducción vuelve a ejecutar los nodos; no se limita a leer desde la caché. Las llamadas a LLM, las solicitudes a la API y las interrupciones se ejecutan de nuevo y pueden devolver resultados diferentes. Reproducir desde el último punto de control (sin nextnodos) no tiene ningún efecto.
Repetición
Utilice esta función get_state_historypara encontrar el punto de control desde el que desea reproducir la partida y, a continuación, llámela invokecon la configuración de ese punto de control:

```python
from langgraph.graph import StateGraph, START
from langgraph.checkpoint.memory import InMemorySaver
from typing_extensions import TypedDict, NotRequired
from langchain_core.utils.uuid import uuid7

class State(TypedDict):
    topic: NotRequired[str]
    joke: NotRequired[str]


def generate_topic(state: State):
    return {"topic": "socks in the dryer"}


def write_joke(state: State):
    return {"joke": f"Why do {state['topic']} disappear? They elope!"}


checkpointer = InMemorySaver()
graph = (
    StateGraph(State)
    .add_node("generate_topic", generate_topic)
    .add_node("write_joke", write_joke)
    .add_edge(START, "generate_topic")
    .add_edge("generate_topic", "write_joke")
    .compile(checkpointer=checkpointer)
)

# Step 1: Run the graph
config = {"configurable": {"thread_id": str(uuid7())}}
result = graph.invoke({}, config)

# Step 2: Find a checkpoint to replay from
history = list(graph.get_state_history(config))
# History is in reverse chronological order
for state in history:
    print(f"next={state.next}, checkpoint_id={state.config['configurable']['checkpoint_id']}")

# Step 3: Replay from a specific checkpoint
# Find the checkpoint before write_joke
before_joke = next(s for s in history if s.next == ("write_joke",))
replay_result = graph.invoke(None, before_joke.config)
# write_joke re-executes (runs again), generate_topic does not
​```

Tenedor
Fork crea una nueva rama a partir de un punto de control anterior con un estado modificado. Llama update_statea un punto de control anterior para crear la bifurcación y luego invokecontinúa Nonela ejecución.
Tenedor
update_stateNo revierte un hilo. Crea un nuevo punto de control que se bifurca desde el punto especificado. El historial de ejecución original permanece intacto .

```python
# Find checkpoint before write_joke
history = list(graph.get_state_history(config))
before_joke = next(s for s in history if s.next == ("write_joke",))

# Fork: update state to change the topic
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
)

# Resume from the fork — write_joke re-executes with the new topic
fork_result = graph.invoke(None, fork_config)
print(fork_result["joke"])  # A joke about chickens, not socks
​```

Desde un nodo específico
Al llamar a esta función update_state, los valores se aplican utilizando los escritores del nodo especificado (incluidos los reductores ). El punto de control registra que ese nodo produjo la actualización y la ejecución se reanuda desde los sucesores de ese nodo.
Por defecto, LangGraph infiere as_nodela versión a partir del historial de versiones del punto de control. Al bifurcar desde un punto de control específico, esta inferencia casi siempre es correcta.
Especifique as_nodeexplícitamente cuándo:
Ramas paralelas : Varios nodos actualizaron su estado en el mismo paso, y LangGraph no puede determinar cuál fue el último ( InvalidUpdateError).
Sin historial de ejecución : Configurando el estado en un hilo nuevo (común en las pruebas ).
Saltar nodos : Configúrelo as_nodepara que se ejecute en un nodo posterior para que el gráfico piense que ese nodo ya se ha ejecutado.

```python
# graph: generate_topic -> write_joke

# Treat this update as if generate_topic produced it.
# Execution resumes at write_joke (the successor of generate_topic).
fork_config = graph.update_state(
    before_joke.config,
    values={"topic": "chickens"},
    as_node="generate_topic",
)
​```

Interrumpe
Si su gráfico utiliza interruptflujos de trabajo con intervención humana , las interrupciones siempre se vuelven a activar durante el viaje en el tiempo. El nodo que contiene la interrupción se vuelve a ejecutar y interrupt()se detiene para un nuevo Command(resume=...).

```python
from langgraph.types import interrupt, Command

class State(TypedDict):
    value: list[str]

def ask_human(state: State):
    answer = interrupt("What is your name?")
    return {"value": [f"Hello, {answer}!"]}

def final_step(state: State):
    return {"value": ["Done"]}

graph = (
    StateGraph(State)
    .add_node("ask_human", ask_human)
    .add_node("final_step", final_step)
    .add_edge(START, "ask_human")
    .add_edge("ask_human", "final_step")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# First run: hits interrupt
graph.invoke({"value": []}, config)
# Resume with answer
graph.invoke(Command(resume="Alice"), config)

# Replay from before ask_human
history = list(graph.get_state_history(config))
before_ask = [s for s in history if s.next == ("ask_human",)][-1]

replay_result = graph.invoke(None, before_ask.config)
# Pauses at interrupt — waiting for new Command(resume=...)

# Fork from before ask_human
fork_config = graph.update_state(before_ask.config, {"value": ["forked"]})
fork_result = graph.invoke(None, fork_config)
# Pauses at interrupt — waiting for new Command(resume=...)

# Resume the forked interrupt with a different answer
graph.invoke(Command(resume="Bob"), fork_config)
# Result: {"value": ["forked", "Hello, Bob!", "Done"]}
​
Múltiples interrupciones
Si su gráfico recopila información en varios puntos (por ejemplo, un formulario de varios pasos), puede bifurcar el proceso entre las interrupciones para cambiar una respuesta posterior sin volver a formular las preguntas anteriores.
def ask_name(state):
    name = interrupt("What is your name?")
    return {"value": [f"name:{name}"]}

def ask_age(state):
    age = interrupt("How old are you?")
    return {"value": [f"age:{age}"]}

# Graph: ask_name -> ask_age -> final
# After completing both interrupts:

# Fork from BETWEEN the two interrupts (after ask_name, before ask_age)
history = list(graph.get_state_history(config))
between = [s for s in history if s.next == ("ask_age",)][-1]

fork_config = graph.update_state(between.config, {"value": ["modified"]})
result = graph.invoke(None, fork_config)
# ask_name result preserved ("name:Alice")
# ask_age pauses at interrupt — waiting for new answer
​```

Subgrafos
El viaje en el tiempo con subgrafos depende de si el subgrafo tiene su propio punto de control. Esto determina la granularidad de los puntos de control desde los que se puede viajar en el tiempo.
Punto de control heredado (predeterminado)
puntero de control de subgrafo
Por defecto, un subgrafo hereda el punto de control del padre. El padre trata todo el subgrafo como un único superpaso ; solo existe un punto de control a nivel del padre para toda la ejecución del subgrafo. Si se viaja en el tiempo desde antes de que el subgrafo se vuelva a ejecutar desde cero, se vuelve a ejecutar.
No se puede viajar en el tiempo a un punto entre nodos en un subgrafo predeterminado; solo se puede viajar en el tiempo desde el nivel padre.

```python
# Subgraph without its own checkpointer (default)
subgraph = (
    StateGraph(State)
    .add_node("step_a", step_a)       # Has interrupt()
    .add_node("step_b", step_b)       # Has interrupt()
    .add_edge(START, "step_a")
    .add_edge("step_a", "step_b")
    .compile()  # No checkpointer — inherits from parent
)

graph = (
    StateGraph(State)
    .add_node("subgraph_node", subgraph)
    .add_edge(START, "subgraph_node")
    .compile(checkpointer=InMemorySaver())
)

config = {"configurable": {"thread_id": "1"}}

# Complete both interrupts
graph.invoke({"value": []}, config)            # Hits step_a interrupt
graph.invoke(Command(resume="Alice"), config)  # Hits step_b interrupt
graph.invoke(Command(resume="30"), config)     # Completes

# Time travel from before the subgraph
history = list(graph.get_state_history(config))
before_sub = [s for s in history if s.next == ("subgraph_node",)][-1]

fork_config = graph.update_state(before_sub.config, {"value": ["forked"]})
result = graph.invoke(None, fork_config)
# The entire subgraph re-executes from scratch
# You cannot time travel to a point between step_a and step_b
```

### Memoria

Las aplicaciones de IA necesitan memoria para compartir el contexto entre múltiples interacciones. En LangGraph, puedes agregar dos tipos de memoria:
Añade memoria a corto plazo como parte del estado de tu agente para habilitar conversaciones de varios turnos.
Agregue memoria a largo plazo para almacenar datos específicos del usuario o a nivel de aplicación entre sesiones.
​
Añadir memoria a corto plazo
La memoria a corto plazo ( persistencia a nivel de hilo ) permite a los agentes realizar un seguimiento de conversaciones de varios turnos. Para agregar memoria a corto plazo:

```python
from langgraph.checkpoint.memory import InMemorySaver  
from langgraph.graph import StateGraph

checkpointer = InMemorySaver()

builder = StateGraph(...)
graph = builder.compile(checkpointer=checkpointer)

graph.invoke(
    {"messages": [{"role": "user", "content": "hi! i am Bob"}]},
    {"configurable": {"thread_id": "1"}},
)
​```


Uso en producción
En producción, utilice un sistema de puntos de control respaldado por una base de datos:
from langgraph.checkpoint.postgres import PostgresSaver

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    builder = StateGraph(...)
    graph = builder.compile(checkpointer=checkpointer)

#### Ejemplo: uso de un punto de control de Postgres

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

Debes llamar a `checkpointer.setup()` la primera vez que uses Postgres.

**Sincronizar** | **Asíncrono**

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres import PostgresSaver  

model = init_chat_model(model="claude-haiku-4-5-20251001")

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
```

#### Ejemplo: uso de un punto de control de MongoDB

```bash
pip install -U pymongo langgraph langgraph-checkpoint-mongodb
```

**Configuración**: Para usar el punto de control de MongoDB, necesitará un clúster de MongoDB. Siga esta guía para crear un clúster si aún no tiene uno.

**Sincronizar** | **Asíncrono**

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.mongodb import MongoDBSaver  

model = init_chat_model(model="claude-haiku-4-5-20251001")

DB_URI = "localhost:27017"
with MongoDBSaver.from_conn_string(DB_URI) as checkpointer:

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
```

#### Ejemplo: uso de Redis checkpointer

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

Debes llamar a `checkpointer.setup()` la primera vez que uses el punto de control de Redis.

**Sincronizar** | **Asíncrono**

```python
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis import RedisSaver  

model = init_chat_model(model="claude-haiku-4-5-20251001")

DB_URI = "redis://localhost:6379"
with RedisSaver.from_conn_string(DB_URI) as checkpointer:
    # checkpointer.setup()

    def call_model(state: MessagesState):
        response = model.invoke(state["messages"])
        return {"messages": response}

    builder = StateGraph(MessagesState)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(checkpointer=checkpointer)

    config = {
        "configurable": {
            "thread_id": "1"
        }
    }

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()

    for chunk in graph.stream(
        {"messages": [{"role": "user", "content": "what's my name?"}]},
        config,
        stream_mode="values"
    ):
        chunk["messages"][-1].pretty_print()
​```

Uso en subgrafos
Si su grafo contiene subgrafos , solo necesita proporcionar el puntero de control al compilar el grafo padre. LangGraph propagará automáticamente el puntero de control a los subgrafos hijos.

```python
from langgraph.graph import START, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict

class State(TypedDict):
    foo: str

# Subgraph

def subgraph_node_1(state: State):
    return {"foo": state["foo"] + "bar"}

subgraph_builder = StateGraph(State)
subgraph_builder.add_node(subgraph_node_1)
subgraph_builder.add_edge(START, "subgraph_node_1")
subgraph = subgraph_builder.compile()

# Parent graph

builder = StateGraph(State)
builder.add_node("node_1", subgraph)
builder.add_edge(START, "node_1")

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)
```

Puede configurar el comportamiento de los puntos de control específicos para cada subgrafo. Consulte la sección sobre persistencia de subgrafos para obtener más información sobre los niveles de persistencia, incluido el soporte para interrupciones y las continuaciones con estado.
```python
subgraph_builder = StateGraph(...)
subgraph = subgraph_builder.compile(checkpointer=True)
​```
Añadir memoria a largo plazo
Utilice la memoria a largo plazo para almacenar datos específicos del usuario o de la aplicación a lo largo de las conversaciones.

```python
from langgraph.store.memory import InMemoryStore  
from langgraph.graph import StateGraph

store = InMemoryStore()

builder = StateGraph(...)
graph = builder.compile(store=store)
​```

Acceda a la tienda dentro de los nodos
Una vez que compile un grafo con un almacén, LangGraph inyecta automáticamente el almacén en las funciones de su nodo. La forma recomendada de acceder al almacén es a través del Runtimeobjeto.

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime
from langgraph.graph import StateGraph, MessagesState, START
import uuid

@dataclass
class Context:
    user_id: str

async def call_model(state: MessagesState, runtime: Runtime[Context]):
    user_id = runtime.context.user_id  
    namespace = (user_id, "memories")

    # Search for relevant memories
    memories = await runtime.store.asearch(
        namespace, query=state["messages"][-1].content, limit=3
    )
    info = "\n".join([d.value["data"] for d in memories])

    # ... Use memories in model call

    # Store a new memory
    await runtime.store.aput(
        namespace, str(uuid.uuid4()), {"data": "User prefers dark mode"}
    )

builder = StateGraph(MessagesState, context_schema=Context)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(store=store)

# Pass context at invocation time
graph.invoke(
    {"messages": [{"role": "user", "content": "hi"}]},
    {"configurable": {"thread_id": "1"}},
    context=Context(user_id="1"),
)
​```

Uso en producción
En producción, utilice un almacén respaldado por una base de datos:

```python
from langgraph.store.postgres import PostgresStore

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"
with PostgresStore.from_conn_string(DB_URI) as store:
    builder = StateGraph(...)
    graph = builder.compile(store=store)
```

#### Ejemplo: uso del almacén Postgres

```bash
pip install -U "psycopg[binary,pool]" langgraph langgraph-checkpoint-postgres
```

Debes llamar a `store.setup()` la primera vez que uses el almacén de Postgres.

**Asíncrono** | **Sincronizar**

```python
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.store.postgres.aio import AsyncPostgresStore  
from langgraph.runtime import Runtime  
import uuid

model = init_chat_model(model="claude-haiku-4-5-20251001")

@dataclass
class Context:
    user_id: str

async def call_model(
    state: MessagesState,
    runtime: Runtime[Context],
):
    user_id = runtime.context.user_id  
    namespace = ("memories", user_id)
    memories = await runtime.store.asearch(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        await runtime.store.aput(namespace, str(uuid.uuid4()), {"data": memory})

    response = await model.ainvoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}

DB_URI = "postgresql://postgres:postgres@localhost:5442/postgres?sslmode=disable"

async with (
    AsyncPostgresStore.from_conn_string(DB_URI) as store,
    AsyncPostgresSaver.from_conn_string(DB_URI) as checkpointer,
):
    # await store.setup()
    # await checkpointer.setup()

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    config = {"configurable": {"thread_id": "1"}}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
        config,
        stream_mode="values",
        context=Context(user_id="1"),
    ):
        chunk["messages"][-1].pretty_print()

    config = {"configurable": {"thread_id": "2"}}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "what is my name?"}]},
        config,
        stream_mode="values",
        context=Context(user_id="1"),
    ):
        chunk["messages"][-1].pretty_print()
```

#### Ejemplo: uso del almacén Redis

```bash
pip install -U langgraph langgraph-checkpoint-redis
```

Debes llamar a `store.setup()` la primera vez que uses el almacén Redis.

**Asíncrono** | **Sincronizar**

```python
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.checkpoint.redis.aio import AsyncRedisSaver
from langgraph.store.redis.aio import AsyncRedisStore  
from langgraph.runtime import Runtime  
import uuid

model = init_chat_model(model="claude-haiku-4-5-20251001")

@dataclass
class Context:
    user_id: str

async def call_model(
    state: MessagesState,
    runtime: Runtime[Context],
):
    user_id = runtime.context.user_id  
    namespace = ("memories", user_id)
    memories = await runtime.store.asearch(namespace, query=str(state["messages"][-1].content))
    info = "\n".join([d.value["data"] for d in memories])
    system_msg = f"You are a helpful assistant talking to the user. User info: {info}"

    # Store new memories if the user asks the model to remember
    last_message = state["messages"][-1]
    if "remember" in last_message.content.lower():
        memory = "User name is Bob"
        await runtime.store.aput(namespace, str(uuid.uuid4()), {"data": memory})

    response = await model.ainvoke(
        [{"role": "system", "content": system_msg}] + state["messages"]
    )
    return {"messages": response}

DB_URI = "redis://localhost:6379"

async with (
    AsyncRedisStore.from_conn_string(DB_URI) as store,
    AsyncRedisSaver.from_conn_string(DB_URI) as checkpointer,
):
    # await store.setup()
    # await checkpointer.asetup()

    builder = StateGraph(MessagesState, context_schema=Context)
    builder.add_node(call_model)
    builder.add_edge(START, "call_model")

    graph = builder.compile(
        checkpointer=checkpointer,
        store=store,
    )

    config = {"configurable": {"thread_id": "1"}}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "Hi! Remember: my name is Bob"}]},
        config,
        stream_mode="values",
        context=Context(user_id="1"),
    ):
        chunk["messages"][-1].pretty_print()

    config = {"configurable": {"thread_id": "2"}}
    async for chunk in graph.astream(
        {"messages": [{"role": "user", "content": "what is my name?"}]},
        config,
        stream_mode="values",
        context=Context(user_id="1"),
    ):
        chunk["messages"][-1].pretty_print()
​```

Utilice la búsqueda semántica
Habilita la búsqueda semántica en el almacén de memoria de tu grafo para permitir que los agentes del grafo busquen elementos en el almacén por similitud semántica.

```python
from langchain.embeddings import init_embeddings
from langgraph.store.memory import InMemoryStore

# Create store with semantic search enabled
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

items = store.search(
    ("user_123", "memories"), query="I'm hungry", limit=1
)
Memoria a largo plazo con búsqueda semántica


from langchain.embeddings import init_embeddings
from langchain.chat_models import init_chat_model
from langgraph.store.memory import InMemoryStore
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.runtime import Runtime  

model = init_chat_model("gpt-4.1-mini")

# Create store with semantic search enabled
embeddings = init_embeddings("openai:text-embedding-3-small")
store = InMemoryStore(
    index={
        "embed": embeddings,
        "dims": 1536,
    }
)

store.put(("user_123", "memories"), "1", {"text": "I love pizza"})
store.put(("user_123", "memories"), "2", {"text": "I am a plumber"})

async def chat(state: MessagesState, runtime: Runtime):
    # Search based on user's last message
    items = await runtime.store.asearch(
        ("user_123", "memories"), query=state["messages"][-1].content, limit=2
    )
    memories = "\n".join(item.value["text"] for item in items)
    memories = f"## Memories of user\n{memories}" if memories else ""
    response = await model.ainvoke(
        [
            {"role": "system", "content": f"You are a helpful assistant.\n{memories}"},
            *state["messages"],
        ]
    )
    return {"messages": [response]}


builder = StateGraph(MessagesState)
builder.add_node(chat)
builder.add_edge(START, "chat")
graph = builder.compile(store=store)

async for message, metadata in graph.astream(
    input={"messages": [{"role": "user", "content": "I'm hungry"}]},
    stream_mode="messages",
):
    print(message.content, end="")
​```

## Gestionar la memoria a corto plazo

Con la memoria a corto plazo habilitada, las conversaciones largas pueden exceder la ventana de contexto del LLM. Las soluciones comunes son:

- **Recortar mensajes**: Eliminar los primeros o los últimos N mensajes (antes de llamar a LLM).
- **Eliminar mensajes del estado de LangGraph** de forma permanente.
- **Resumir mensajes**: Resume los mensajes anteriores del historial y reemplázalos con un resumen.
- **Gestiona los puntos de control** para almacenar y recuperar el historial de mensajes.
- **Estrategias personalizadas** (por ejemplo, filtrado de mensajes, etc.).

Esto permite al agente realizar un seguimiento de la conversación sin exceder la ventana de contexto del LLM.

### Recortar mensajes

La mayoría de los LLM tienen una ventana de contexto máxima admitida (expresada en tokens). Una forma de decidir cuándo truncar los mensajes es contar los tokens en el historial del mensaje y truncarlos cuando se acerque a ese límite. Si utiliza LangChain, puede usar la utilidad `trim_messages` y especificar la cantidad de tokens que desea conservar de la lista, así como el valor `strategy` (por ejemplo, conservar el último `max_tokens`) que se utilizará para gestionar el límite.

Para recortar el historial de mensajes, utilice la función `trim_messages`:

```python
from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately  
)

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node(call_model)
...
Ejemplo completo: recortar mensajes

from langchain_core.messages.utils import (
    trim_messages,
    count_tokens_approximately  
)
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState

model = init_chat_model("claude-sonnet-4-6")
summarization_model = model.bind(max_tokens=128)

def call_model(state: MessagesState):
    messages = trim_messages(
        state["messages"],
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=128,
        start_on="human",
        end_on=("human", "tool"),
    )
    response = model.invoke(messages)
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(MessagesState)
builder.add_node(call_model)
builder.add_edge(START, "call_model")
graph = builder.compile(checkpointer=checkpointer)

config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
final_response = graph.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
================================== Ai Message ==================================

Your name is Bob, as you mentioned when you first introduced yourself.
​```

### Eliminar mensajes

Puedes eliminar mensajes del estado del gráfico para gestionar el historial de mensajes. Esto resulta útil cuando deseas eliminar mensajes específicos o borrar todo el historial de mensajes.

Para eliminar mensajes del estado del gráfico, puede usar el `RemoveMessage`. Para que `RemoveMessage` funcione, necesita usar una clave de estado con reductor, como `add_messages` en `MessagesState`.

Para eliminar mensajes específicos:

```python
from langchain.messages import RemoveMessage

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}
```

Para eliminar todos los mensajes:

```python
from langgraph.graph.message import REMOVE_ALL_MESSAGES

def delete_messages(state):
    return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}
```

Al eliminar mensajes, asegúrese de que el historial de mensajes resultante sea válido. Verifique las limitaciones del proveedor LLM que está utilizando. Por ejemplo:
Algunos proveedores esperan que el historial de mensajes comience con un usermensaje.
La mayoría de los proveedores requieren assistantque los mensajes con llamadas a herramientas vayan seguidos de toollos mensajes de resultados correspondientes.
Ejemplo completo: eliminar mensajes

```python
from langchain.messages import RemoveMessage  

def delete_messages(state):
    messages = state["messages"]
    if len(messages) > 2:
        # remove the earliest two messages
        return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}

builder = StateGraph(MessagesState)
builder.add_sequence([call_model, delete_messages])
builder.add_edge(START, "call_model")

checkpointer = InMemorySaver()
app = builder.compile(checkpointer=checkpointer)

for event in app.stream(
    {"messages": [{"role": "user", "content": "hi! I'm bob"}]},
    config,
    stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])

for event in app.stream(
    {"messages": [{"role": "user", "content": "what's my name?"}]},
    config,
    stream_mode="values"
):
    print([(message.type, message.content) for message in event["messages"]])
[('human', "hi! I'm bob")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?')]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?'), ('human', "what's my name?")]
[('human', "hi! I'm bob"), ('ai', 'Hi Bob! How are you doing today? Is there anything I can help you with?'), ('human', "what's my name?"), ('ai', 'Your name is Bob.')]
[('human', "what's my name?"), ('ai', 'Your name is Bob.')]
​```

### Resumir mensajes

El problema de recortar o eliminar mensajes, como se muestra arriba, es que se puede perder información al depurar la cola de mensajes. Por ello, algunas aplicaciones se benefician de un enfoque más sofisticado que resume el historial de mensajes mediante un modelo de chat.

#### Resumen

La lógica de solicitud y orquestación se puede utilizar para resumir el historial de mensajes. Por ejemplo, en LangGraph se puede extender `MessagesState` para incluir una clave `summary`:

```python
from langgraph.graph import MessagesState
class State(MessagesState):
    summary: str
```

Luego, puedes generar un resumen del historial de chat, utilizando cualquier resumen existente como contexto para el siguiente. Este nodo `summarize_conversation` se puede llamar después de que se haya acumulado cierta cantidad de mensajes en la clave de estado `messages`.

```python
def summarize_conversation(state: State):

    # First, we get any existing summary
    summary = state.get("summary", "")

    # Create our summarization prompt
    if summary:

        # A summary already exists
        summary_message = (
            f"This is a summary of the conversation to date: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )

    else:
        summary_message = "Create a summary of the conversation above:"

    # Add prompt to our history
    messages = state["messages"] + [HumanMessage(content=summary_message)]
    response = model.invoke(messages)

    # Delete all but the 2 most recent messages
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    return {"summary": response.content, "messages": delete_messages}
Ejemplo completo: resumir mensajes

from typing import Any, TypedDict

from langchain.chat_models import init_chat_model
from langchain.messages import AnyMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.graph import StateGraph, START, MessagesState
from langgraph.checkpoint.memory import InMemorySaver
from langmem.short_term import SummarizationNode, RunningSummary  

model = init_chat_model("claude-sonnet-4-6")
summarization_model = model.bind(max_tokens=128)

class State(MessagesState):
    context: dict[str, RunningSummary]

class LLMInputState(TypedDict):
    summarized_messages: list[AnyMessage]
    context: dict[str, RunningSummary]

summarization_node = SummarizationNode(
    token_counter=count_tokens_approximately,
    model=summarization_model,
    max_tokens=256,
    max_tokens_before_summary=256,
    max_summary_tokens=128,
)

def call_model(state: LLMInputState):
    response = model.invoke(state["summarized_messages"])
    return {"messages": [response]}

checkpointer = InMemorySaver()
builder = StateGraph(State)
builder.add_node(call_model)
builder.add_node("summarize", summarization_node)
builder.add_edge(START, "summarize")
builder.add_edge("summarize", "call_model")
graph = builder.compile(checkpointer=checkpointer)

# Invoke the graph
config = {"configurable": {"thread_id": "1"}}
graph.invoke({"messages": "hi, my name is bob"}, config)
graph.invoke({"messages": "write a short poem about cats"}, config)
graph.invoke({"messages": "now do the same but for dogs"}, config)
final_response = graph.invoke({"messages": "what's my name?"}, config)

final_response["messages"][-1].pretty_print()
print("\nSummary:", final_response["context"]["running_summary"].summary)
```

Haremos un seguimiento de nuestro resumen de carrera en el contextcampo.
(esperado por el SummarizationNode).
Defina un estado privado que se utilizará únicamente para filtrar.
las entradas al call_modelnodo.
Estamos pasando un estado de entrada privado aquí para aislar los mensajes devueltos por el nodo de resumen.
================================== Ai Message ==================================

From our conversation, I can see that you introduced yourself as Bob. That's the name you shared with me when we began talking.

Summary: In this conversation, I was introduced to Bob, who then asked me to write a poem about cats. I composed a poem titled "The Mystery of Cats" that captured cats' graceful movements, independent nature, and their special relationship with humans. Bob then requested a similar poem about dogs, so I wrote "The Joy of Dogs," which highlighted dogs' loyalty, enthusiasm, and loving companionship. Both poems were written in a similar style but emphasized the distinct characteristics that make each pet special.
​
Gestionar puntos de control
Puedes ver y eliminar la información almacenada por el punto de control.
​
Ver estado del hilo
API gráfica/funcional
API de Checkpointer

```python
config = {
    "configurable": {
        "thread_id": "1",
        # optionally provide an ID for a specific checkpoint,
        # otherwise the latest checkpoint is shown
        # "checkpoint_id": "1f029ca3-1f5b-6704-8004-820c16b69a5a"  #

    }
}
graph.get_state(config)
StateSnapshot(
    values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today?), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]}, next=(),
    config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
    metadata={
        'source': 'loop',
        'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}},
        'step': 4,
        'parents': {},
        'thread_id': '1'
    },
    created_at='2025-05-05T16:01:24.680462+00:00',
    parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
    tasks=(),
    interrupts=()
)
​```

Ver el historial del hilo
API gráfica/funcional
API de Checkpointer

```python
config = {
    "configurable": {
        "thread_id": "1"
    }
}
list(graph.get_state_history(config))
[
    StateSnapshot(
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?"), AIMessage(content='Your name is Bob.')]},
        next=(),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1f5b-6704-8004-820c16b69a5a'}},
        metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Your name is Bob.')}}, 'step': 4, 'parents': {}, 'thread_id': '1'},
        created_at='2025-05-05T16:01:24.680462+00:00',
        parent_config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
        tasks=(),
        interrupts=()
    ),
    StateSnapshot(
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?'), HumanMessage(content="what's my name?")]},
        next=('call_model',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-1790-6b0a-8003-baf965b6a38f'}},
        metadata={'source': 'loop', 'writes': None, 'step': 3, 'parents': {}, 'thread_id': '1'},
        created_at='2025-05-05T16:01:23.863421+00:00',
        parent_config={...}
        tasks=(PregelTask(id='8ab4155e-6b15-b885-9ce5-bed69a2c305c', name='call_model', path=('__pregel_pull', 'call_model'), error=None, interrupts=(), state=None, result={'messages': AIMessage(content='Your name is Bob.')}),),
        interrupts=()
    ),
    StateSnapshot(
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]},
        next=('__start__',),
        config={...},
        metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "what's my name?"}]}}, 'step': 2, 'parents': {}, 'thread_id': '1'},
        created_at='2025-05-05T16:01:23.863173+00:00',
        parent_config={...}
        tasks=(PregelTask(id='24ba39d6-6db1-4c9b-f4c5-682aeaf38dcd', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': "what's my name?"}]}),),
        interrupts=()
    ),
    StateSnapshot(
        values={'messages': [HumanMessage(content="hi! I'm bob"), AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')]},
        next=(),
        config={...},
        metadata={'source': 'loop', 'writes': {'call_model': {'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}}, 'step': 1, 'parents': {}, 'thread_id': '1'},
        created_at='2025-05-05T16:01:23.862295+00:00',
        parent_config={...}
        tasks=(),
        interrupts=()
    ),
    StateSnapshot(
        values={'messages': [HumanMessage(content="hi! I'm bob")]},
        next=('call_model',),
        config={...},
        metadata={'source': 'loop', 'writes': None, 'step': 0, 'parents': {}, 'thread_id': '1'},
        created_at='2025-05-05T16:01:22.278960+00:00',
        parent_config={...}
        tasks=(PregelTask(id='8cbd75e0-3720-b056-04f7-71ac805140a0', name='call_model', path=('__pregel_pull', 'call_model'), error=None, interrupts=(), state=None, result={'messages': AIMessage(content='Hi Bob! How are you doing today? Is there anything I can help you with?')}),),
        interrupts=()
    ),
    StateSnapshot(
        values={'messages': []},
        next=('__start__',),
        config={'configurable': {'thread_id': '1', 'checkpoint_ns': '', 'checkpoint_id': '1f029ca3-0870-6ce2-bfff-1f3f14c3e565'}},
        metadata={'source': 'input', 'writes': {'__start__': {'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}}, 'step': -1, 'parents': {}, 'thread_id': '1'},
        created_at='2025-05-05T16:01:22.277497+00:00',
        parent_config=None,
        tasks=(PregelTask(id='d458367b-8265-812c-18e2-33001d199ce6', name='__start__', path=('__pregel_pull', '__start__'), error=None, interrupts=(), state=None, result={'messages': [{'role': 'user', 'content': "hi! I'm bob"}]}),),
        interrupts=()
    )
]
​```

Eliminar todos los puntos de control de un hilo
thread_id = "1"
checkpointer.delete_thread(thread_id)
​
### Gestión de bases de datos

Si utiliza alguna implementación de persistencia basada en bases de datos (como Postgres o Redis) para almacenar memoria a corto y/o largo plazo, deberá ejecutar migraciones para configurar el esquema necesario antes de poder utilizarlo con su base de datos.

Por convención, la mayoría de las bibliotecas específicas de bases de datos definen un método `setup()` en el puntero de control o la instancia de almacenamiento que ejecuta las migraciones necesarias. Sin embargo, conviene consultar la implementación específica `BaseCheckpointSaver` para `BaseStore` para confirmar el nombre exacto del método y su uso.

Recomendamos ejecutar las migraciones como un paso de implementación específico, o bien, puede asegurarse de que se ejecuten como parte del inicio del servidor.

---

## Crea un agente RAG personalizado con LangGraph

### Descripción general

En este tutorial construiremos un agente de recuperación utilizando LangGraph.

LangChain ofrece implementaciones de agentes integradas, implementadas mediante primitivas de LangGraph. Si se requiere una personalización más profunda, los agentes pueden implementarse directamente en LangGraph. Esta guía muestra un ejemplo de implementación de un agente de recuperación. Los agentes de recuperación son útiles cuando se desea que un LLM decida si recuperar el contexto de un almacén de vectores o responder directamente al usuario.

Al finalizar el tutorial habremos hecho lo siguiente:

1. Obtener y preprocesar los documentos que se utilizarán para su recuperación.
2. Indexar esos documentos para la búsqueda semántica y crear una herramienta de recuperación para el agente.
3. Construir un sistema RAG con capacidad de decisión sobre cuándo utilizar la herramienta de recuperación.

#### RAG híbrido

### Conceptos

Abordaremos los siguientes conceptos:

- Recuperación mediante cargadores de documentos, divisores de texto, incrustaciones y almacenes vectoriales.
- La API de grafos de LangGraph, que incluye estado, nodos, aristas y aristas condicionales.
### Configuración

Vamos a descargar los paquetes necesarios y configurar nuestras claves API:

```bash
pip install -U langgraph "langchain[openai]" langchain-community langchain-text-splitters bs4
```

```python
import getpass
import os
```


def _set_env(key: str):
    if key not in os.environ:
        os.environ[key] = getpass.getpass(f"{key}:")


_set_env("OPENAI_API_KEY")
Regístrate en LangSmith para detectar rápidamente problemas y mejorar el rendimiento de tus proyectos LangGraph. LangSmith te permite usar datos de seguimiento para depurar, probar y supervisar tus aplicaciones LLM creadas con LangGraph.
​
1. Preprocesar documentos
Obtenga los documentos que utilizaremos en nuestro sistema RAG. Usaremos tres de las páginas más recientes del excelente blog de Lilian Weng . Comenzaremos obteniendo el contenido de las páginas usando WebBaseLoaderla utilidad:
from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://lilianweng.github.io/posts/2024-11-28-reward-hacking/",
    "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
    "https://lilianweng.github.io/posts/2024-04-12-diffusion-video/",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs[0][0].page_content.strip()[:1000]
Dividir los documentos recuperados en fragmentos más pequeños para su indexación en nuestro almacén de vectores:
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs_list = [item for sublist in docs for item in sublist]

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=100, chunk_overlap=50
)
doc_splits = text_splitter.split_documents(docs_list)
doc_splits[0].page_content.strip()
​
2. Crear una herramienta de recuperación
Ahora que tenemos nuestros documentos divididos, podemos indexarlos en un almacén vectorial que utilizaremos para la búsqueda semántica.
Utilice un almacén de vectores en memoria y las incrustaciones de OpenAI:
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

vectorstore = InMemoryVectorStore.from_documents(
    documents=doc_splits, embedding=OpenAIEmbeddings()
)
retriever = vectorstore.as_retriever()
Crea una herramienta de recuperación usando el @tooldecorador:
from langchain.tools import tool

@tool
def retrieve_blog_posts(query: str) -> str:
    """Search and return information about Lilian Weng blog posts."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

retriever_tool = retrieve_blog_posts
Prueba la herramienta:
retriever_tool.invoke({"query": "types of reward hacking"})
​
3. Generar consulta
Ahora comenzaremos a construir componentes ( nodos y aristas ) para nuestro grafo RAG agencial.
Tenga en cuenta que los componentes operarán sobre el MessagesStateestado del gráfico que contiene una messagesclave con una lista de mensajes de chat .
Construye un generate_query_or_respondnodo. Este llamará a un LLM para generar una respuesta basada en el estado actual del grafo (lista de mensajes). Dados los mensajes de entrada, decidirá recuperarlos usando la herramienta de recuperación o responder directamente al usuario. Ten en cuenta que le estamos dando al modelo de chat acceso a lo que retriever_toolcreamos anteriormente a través de .bind_tools:
from langgraph.graph import MessagesState
from langchain.chat_models import init_chat_model

response_model = init_chat_model("gpt-4.1", temperature=0)


def generate_query_or_respond(state: MessagesState):
    """Call the model to generate a response based on the current state. Given
    the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
    """
    response = (
        response_model
        .bind_tools([retriever_tool]).invoke(state["messages"])
    )
    return {"messages": [response]}
Inténtalo con una entrada aleatoria:
input = {"messages": [{"role": "user", "content": "hello!"}]}
generate_query_or_respond(input)["messages"][-1].pretty_print()
Producción:
================================== Ai Message ==================================

Hello! How can I help you today?
Formula una pregunta que requiera búsqueda semántica:
input = {
    "messages": [
        {
            "role": "user",
            "content": "What does Lilian Weng say about types of reward hacking?",
        }
    ]
}
generate_query_or_respond(input)["messages"][-1].pretty_print()
Producción:
================================== Ai Message ==================================
Tool Calls:
retrieve_blog_posts (call_tYQxgfIlnQUDMdtAhdbXNwIM)
Call ID: call_tYQxgfIlnQUDMdtAhdbXNwIM
Args:
    query: types of reward hacking
​
4. Documentos de calificación
Agregue una arista condicional — grade_documents—para determinar si los documentos recuperados son relevantes para la pregunta. Usaremos un modelo con un esquema de salida estructurado GradeDocumentspara la calificación de documentos. La grade_documentsfunción devolverá el nombre del nodo al que se debe ir según la decisión de calificación ( generate_answero rewrite_question):
from pydantic import BaseModel, Field
from typing import Literal

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)


class GradeDocuments(BaseModel):
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = init_chat_model("gpt-4.1", temperature=0)


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = (
        grader_model
        .with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
    )
    score = response.binary_score

    if score == "yes":
        return "generate_answer"
    else:
        return "rewrite_question"
Ejecute esto con documentos irrelevantes en la respuesta de la herramienta:
from langchain_core.messages import convert_to_messages

input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}
grade_documents(input)
Confirme que los documentos pertinentes estén clasificados como tales:
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}
grade_documents(input)
​
5. Reescribe la pregunta
Construir el rewrite_questionnodo. La herramienta de recuperación puede devolver documentos potencialmente irrelevantes, lo que indica la necesidad de mejorar la pregunta original del usuario. Para ello, llamaremos al rewrite_questionnodo:
from langchain.messages import HumanMessage

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)


def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [HumanMessage(content=response.content)]}
Pruébalo:
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}

response = rewrite_question(input)
print(response["messages"][-1].content)
Producción:
What are the different types of reward hacking described by Lilian Weng, and how does she explain them?
​
6. Genera una respuesta
Construir generate_answernodo: si pasamos las comprobaciones del evaluador, podemos generar la respuesta final basándonos en la pregunta original y el contexto recuperado:
GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}
Pruébalo:
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {
                "role": "tool",
                "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
                "tool_call_id": "1",
            },
        ]
    )
}

response = generate_answer(input)
response["messages"][-1].pretty_print()
Producción:
================================== Ai Message ==================================

Lilian Weng categorizes reward hacking into two types: environment or goal misspecification, and reward tampering. She considers reward hacking as a broad concept that includes both of these categories. Reward hacking occurs when an agent exploits flaws or ambiguities in the reward function to achieve high rewards without performing the intended behaviors.
​
7. Ensambla el gráfico.
Ahora vamos a ensamblar todos los nodos y aristas para formar un grafo completo:
Comencemos con una generate_query_or_respondy determinemos si necesitamos llamarretriever_tool
Ruta al siguiente paso usando tools_condition:
Si generate_query_or_respondse devuelve tool_calls, llame retriever_toolpara recuperar el contexto.
De lo contrario, responda directamente al usuario.
Califica el contenido del documento recuperado según su relevancia para la pregunta ( grade_documents) y la ruta al siguiente paso:
Si no es relevante, reformule la pregunta rewrite_questiony luego vuelva a llamar generate_query_or_respond.
Si procede, proceda a generate_answergenerar la respuesta final utilizando el ToolMessagecontexto del documento recuperado.
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

# Compile
graph = workflow.compile()
Visualiza el gráfico:
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
Gráfico del agente SQL
​
8. Ejecutar el RAG agenteico
Ahora probemos el gráfico completo ejecutándolo con una pregunta:
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            }
        ]
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        update["messages"][-1].pretty_print()
        print("\n\n")
Producción:
Update from node generate_query_or_respond
================================== Ai Message ==================================
Tool Calls:
  retrieve_blog_posts (call_NYu2vq4km9nNNEFqJwefWKu1)
 Call ID: call_NYu2vq4km9nNNEFqJwefWKu1
  Args:
    query: types of reward hacking



Update from node retrieve
================================= Tool Message ==================================
Name: retrieve_blog_posts

(Note: Some work defines reward tampering as a distinct category of misalignment behavior from reward hacking. But I consider reward hacking as a broader concept here.)
At a high level, reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering.

Why does Reward Hacking Exist?#

Pan et al. (2022) investigated reward hacking as a function of agent capabilities, including (1) model size, (2) action space resolution, (3) observation space noise, and (4) training time. They also proposed a taxonomy of three types of misspecified proxy rewards:

Let's Define Reward Hacking#
Reward shaping in RL is challenging. Reward hacking occurs when an RL agent exploits flaws or ambiguities in the reward function to obtain high rewards without genuinely learning the intended behaviors or completing the task as designed. In recent years, several related concepts have been proposed, all referring to some form of reward hacking:



Update from node generate_answer
================================== Ai Message ==================================

Lilian Weng categorizes reward hacking into two types: environment or goal misspecification, and reward tampering. She considers reward hacking as a broad concept that includes both of these categories. Reward hacking occurs when an agent exploits flaws or ambiguities in the reward function to achieve high rewards without performing the intended behaviors.

### Ejecución de gráficos

Los agentes de LangGraph no son cajas negras. Cada grafo se compone de nodos con nombre que se ejecutan en secuencia o en paralelo: clasificar, investigar, analizar y sintetizar. Las tarjetas de ejecución del grafo hacen visible este flujo de trabajo al mostrar una tarjeta para cada nodo, indicando su estado, transmitiendo su contenido en tiempo real y realizando un seguimiento de la finalización a lo largo de todo el proceso. Los usuarios ven exactamente qué está haciendo el agente, en qué paso se encuentra y qué produjo cada paso.

Cómo se asignan los nodos del gráfico a las tarjetas de la interfaz de usuario
Un grafo LangGraph define una serie de nodos, cada uno responsable de una tarea específica. Por ejemplo, un proceso de investigación podría tener:
Clasificar : categorizar la consulta del usuario
Investigación : recopilar información relevante
Analizar : extraer conclusiones de la investigación.
Sintetizar : producir una respuesta final y pulida.
Cada nodo escribe su salida en una clave específica del estado del gráfico. Al asignar estos nombres de nodos y claves de estado a componentes de la interfaz de usuario, se puede crear una representación visual de todo el proceso.

const PIPELINE_NODES = [
  { name: "classify", stateKey: "classification", label: "Classify" },
  { name: "do_research", stateKey: "research", label: "Research" },
  { name: "analyze", stateKey: "analysis", label: "Analyze" },
  { name: "synthesize", stateKey: "synthesis", label: "Synthesize" },
];

const PIPELINE_NODE_NAMES = new Set(PIPELINE_NODES.map((n) => n.name));
​
Configurando useStream
Conecta useStreamcomo de costumbre. Las propiedades clave que usarás son messages (para el enrutamiento de contenido en streaming), values(para las salidas de nodos completadas) y getMessagesMetadata(para identificar qué nodo produjo cada token).
Defina una interfaz TypeScript que coincida con el esquema de estado de su agente y pásela como parámetro de tipo para un useStreamacceso seguro a los valores de estado, incluidas las claves de estado personalizadas para cada nodo de la canalización. En los ejemplos siguientes, reemplace typeof myAgentcon el nombre de su interfaz:

import type { BaseMessage } from "@langchain/core/messages";

interface AgentState {
  messages: BaseMessage[];
  classification: string;
  research: string;
  analysis: string;
  synthesis: string;
}

import { useStream } from "@langchain/react";

const AGENT_URL = "http://localhost:2024";

export function PipelineChat() {
  const stream = useStream<typeof myAgent>({
    apiUrl: AGENT_URL,
    assistantId: "graph_execution_cards",
  });

  return (
    <div>
      <PipelineProgress nodes={PIPELINE_NODES} values={stream.values} />
      <NodeCardList
        nodes={PIPELINE_NODES}
        messages={stream.messages}
        values={stream.values}
        getMetadata={stream.getMessagesMetadata}
      />
    </div>
  );
}
​
Enrutamiento de tokens de transmisión a nodos
A medida que el agente transmite, cada mensaje se anota con metadatos que identifican qué nodo del gráfico lo produjo. Úselo getMessagesMetadatapara extraer el langgraph_nodevalor y enrutar los tokens a la tarjeta correcta:
function getStreamingContent(
  messages: BaseMessage[],
  getMetadata: (msg: BaseMessage) => MessageMetadata | undefined
): Record<string, string> {
  const content: Record<string, string> = {};

  for (const message of messages) {
    if (message.type !== "ai") continue;

    const metadata = getMetadata(message);
    const node = metadata?.streamMetadata?.langgraph_node;

    if (node && PIPELINE_NODE_NAMES.has(node)) {
      content[node] = typeof message.content === "string"
        ? message.content
        : "";
    }
  }

  return content;
}
Esto te proporciona un mapa que relaciona el nombre del nodo con su contenido de transmisión actual. A medida que llegan los tokens, la tarjeta correspondiente se actualiza en tiempo real.
LangGraph configura automáticamente este streamMetadata.langgraph_nodecampo. No necesitas ninguna configuración especial en el servidor. Simplemente transmite los mensajes como de costumbre y los metadatos se incluirán.
​
Determinación del estado del nodo
Cada nodo puede estar en uno de cuatro estados: no iniciado, en transmisión, completado o inactivo. El estado se obtiene de dos fuentes: el mapa de contenido en transmisión (para nodos activos) y stream.values(para nodos completados):
type NodeStatus = "idle" | "streaming" | "complete";

function getNodeStatus(
  node: { name: string; stateKey: string },
  streamingContent: Record<string, string>,
  values: Record<string, unknown>
): NodeStatus {
  if (values?.[node.stateKey]) return "complete";
  if (streamingContent[node.name]) return "streaming";
  return "idle";
}
​
Barra de progreso de la construcción del pipeline
Una barra de progreso horizontal en la parte superior ofrece a los usuarios una visión general de todo el proceso. Cada paso es un segmento etiquetado que se completa a medida que se finalizan los nodos:
function PipelineProgress({
  nodes,
  values,
  streamingContent,
}: {
  nodes: typeof PIPELINE_NODES;
  values: Record<string, unknown>;
  streamingContent: Record<string, string>;
}) {
  return (
    <div className="flex items-center gap-1">
      {nodes.map((node, i) => {
        const status = getNodeStatus(node, streamingContent, values);
        const colors = {
          idle: "bg-gray-200 text-gray-500",
          streaming: "bg-blue-400 text-white animate-pulse",
          complete: "bg-green-500 text-white",
        };

        return (
          <div key={node.name} className="flex items-center">
            <div
              className={`rounded-full px-3 py-1 text-xs font-medium ${colors[status]}`}
            >
              {node.label}
            </div>
            {i < nodes.length - 1 && (
              <div
                className={`mx-1 h-0.5 w-6 ${
                  status === "complete" ? "bg-green-500" : "bg-gray-200"
                }`}
              />
            )}
          </div>
        );
      })}
    </div>
  );
}
​
Creación de componentes NodeCard plegables
Cada nodo tiene su propia tarjeta que muestra el distintivo de estado, el contenido (en streaming o final) y un cuerpo desplegable para salidas largas:
function NodeCard({
  node,
  status,
  streamingContent,
  completedContent,
}: {
  node: { name: string; stateKey: string; label: string };
  status: NodeStatus;
  streamingContent: string | undefined;
  completedContent: unknown;
}) {
  const [collapsed, setCollapsed] = useState(false);

  const displayContent =
    status === "complete"
      ? formatContent(completedContent)
      : streamingContent ?? "";

  const statusBadge = {
    idle: { text: "Waiting", className: "bg-gray-100 text-gray-600" },
    streaming: {
      text: "Running",
      className: "bg-blue-100 text-blue-700 animate-pulse",
    },
    complete: { text: "Done", className: "bg-green-100 text-green-700" },
  };

  const badge = statusBadge[status];

  return (
    <div className="rounded-lg border bg-white shadow-sm">
      <button
        onClick={() => setCollapsed(!collapsed)}
        className="flex w-full items-center justify-between p-4"
      >
        <div className="flex items-center gap-3">
          <h3 className="font-semibold">{node.label}</h3>
          <span
            className={`rounded-full px-2 py-0.5 text-xs font-medium ${badge.className}`}
          >
            {badge.text}
          </span>
        </div>
        <ChevronIcon direction={collapsed ? "down" : "up"} />
      </button>

      {!collapsed && displayContent && (
        <div className="border-t px-4 py-3">
          <div className="prose prose-sm max-w-none">
            {displayContent}
            {status === "streaming" && (
              <span className="inline-block h-4 w-1 animate-pulse bg-blue-500" />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function formatContent(value: unknown): string {
  if (typeof value === "string") return value;
  if (value == null) return "";
  return JSON.stringify(value, null, 2);
}
​
Contenido en streaming frente a contenido finalizado
Cada nodo tiene dos fuentes de contenido, y elegir la correcta es fundamental para una experiencia de usuario fluida:
Fuente	Cuándo usar
streamingContent[node.name]	Mientras el nodo está transmitiendo activamente, contiene tokens a medida que llegan.
stream.values[node.stateKey]	Una vez que el nodo se completa, esto contiene la salida final confirmada.
El patrón es el siguiente: mostrar contenido en streaming para actualizaciones en directo y, una vez finalizado el proceso del nodo, volver al valor del estado confirmado.
for (const node of PIPELINE_NODES) {
  const status = getNodeStatus(node, streamingContent, stream.values);

  const content =
    status === "streaming"
      ? streamingContent[node.name]
      : stream.values?.[node.stateKey];
}
El contenido en streaming puede incluir tokens parciales o formato Markdown que aún no se ha formado por completo. Si renderiza Markdown, asegúrese de que su motor de renderizado gestione correctamente la sintaxis incompleta (por ejemplo, un marcador de negrita sin cerrar **).
​
Reuniéndolo todo
Aquí está la lista completa de tarjetas que combina enrutamiento, detección de estado y representación de tarjetas:
function NodeCardList({
  nodes,
  messages,
  values,
  getMetadata,
}: {
  nodes: typeof PIPELINE_NODES;
  messages: BaseMessage[];
  values: Record<string, unknown>;
  getMetadata: (msg: BaseMessage) => MessageMetadata | undefined;
}) {
  const streamingContent = getStreamingContent(messages, getMetadata);

  return (
    <div className="space-y-3">
      {nodes.map((node) => {
        const status = getNodeStatus(node, streamingContent, values);
        return (
          <NodeCard
            key={node.name}
            node={node}
            status={status}
            streamingContent={streamingContent[node.name]}
            completedContent={values?.[node.stateKey]}
          />
        );
      })}
    </div>
  );
}
​
Casos de uso
Las tarjetas de ejecución de gráficos funcionan bien para cualquier canalización de varios pasos donde la visibilidad es importante:
Procesos de investigación : clasificar → recopilar fuentes → analizar → elaborar un informe
Generación de contenido : esquema → borrador → verificación de datos → edición → publicación
Procesamiento de datos : ingesta → validación → transformación → agregación → exportación
Generación de código : comprender los requisitos → planificar la arquitectura → escribir el código → revisar → probar
Flujos de trabajo de decisión : recopilar contexto → evaluar opciones → puntuar alternativas → recomendar
​
Manejo de tuberías dinámicas
No todos los grafos tienen un conjunto fijo de nodos. Algunas canalizaciones agregan u omiten nodos según la entrada. Para solucionar esto, compruebe qué claves de estado tienen valores:
const activeNodes = PIPELINE_NODES.filter(
  (node) =>
    streamingContent[node.name] ||
    values?.[node.stateKey] ||
    node.name === currentNode
);
Esto garantiza que la interfaz de usuario solo muestre tarjetas para los nodos que sean relevantes para la ejecución actual, evitando tarjetas de marcador de posición vacías.
Si su gráfico tiene ramificaciones condicionales (por ejemplo, omitir "Investigación" para consultas fácticas sencillas), los nodos omitidos nunca aparecerán en el contenido en tiempo real ni en los valores de estado. La barra de progreso de su canalización debería reflejar esto atenuando u ocultando los pasos omitidos.
​
Mejores prácticas
Define los nodos de forma declarativa . Mantén tu PIPELINE_NODESmatriz como una única fuente de información que asocie los nombres de los nodos, las claves de estado y las etiquetas de visualización.
Prioriza el contenido en streaming para los nodos activos . Esto proporciona a los usuarios información inmediata. Solo se recurrirá a los valores de estado confirmados una vez que el nodo haya finalizado.
Colapso automático de nodos completados . En flujos de trabajo largos, el colapso automático de las tarjetas finalizadas permite a los usuarios centrarse en el paso activo.
Muestra el tiempo estimado . Si tienes datos históricos sobre cuánto tarda cada nodo, muestra una estimación del tiempo para que el usuario pueda tener una idea clara de lo que puede pasar.
Agregue un indicador de progreso global . Complemente las tarjetas de cada nodo con una barra de progreso general (por ejemplo, "Paso 2 de 4") en la parte superior de la vista de la canalización.
Gestiona los errores por nodo . Si un nodo falla, muestra el error en su tarjeta sin que se colapse toda la canalización. Otros nodos aún pueden completarse correctamente.

Cómo elegir entre API gráficas y funcionales

LangGraph proporciona dos API diferentes para crear flujos de trabajo de agentes: la API de grafos y la API funcional . Ambas API comparten el mismo entorno de ejecución subyacente y pueden utilizarse juntas en la misma aplicación, pero están diseñadas para diferentes casos de uso y preferencias de desarrollo.
Esta guía le ayudará a comprender cuándo utilizar cada API en función de sus necesidades específicas.
​
Guía de decisión rápida
Utilice la API de Graph cuando lo necesite:
Visualización de flujos de trabajo complejos para depuración y documentación.
Gestión explícita del estado con datos compartidos entre múltiples nodos.
Ramificación condicional con múltiples puntos de decisión
Rutas de ejecución paralelas que necesitan fusionarse posteriormente
Colaboración en equipo donde la representación visual facilita la comprensión.
Utilice la API funcional cuando lo desee:
Cambios mínimos en el código de procedimiento existente.
Flujo de control estándar (if/else, bucles, llamadas a funciones)
Estado con ámbito de función sin gestión de estado explícita
Prototipado rápido con menos código repetitivo
Flujos de trabajo lineales con lógica de ramificación simple.
​
Comparación detallada
​
Cuándo usar la API de Graph
La API de Graph utiliza un enfoque declarativo en el que se definen nodos, aristas y estado compartido para crear una estructura gráfica visual.
1. Árboles de decisión complejos y lógica de ramificación
Cuando su flujo de trabajo tiene múltiples puntos de decisión que dependen de diversas condiciones, la API de Graph hace que estas ramificaciones sean explícitas y fáciles de visualizar.
# Graph API: Clear visualization of decision paths
from langgraph.graph import StateGraph
from typing import TypedDict

class AgentState(TypedDict):
    messages: list
    current_tool: str
    retry_count: int

def should_continue(state):
    if state["retry_count"] > 3:
        return "end"
    elif state["current_tool"] == "search":
        return "process_search"
    else:
        return "call_llm"

workflow = StateGraph(AgentState)
workflow.add_node("call_llm", call_llm_node)
workflow.add_node("process_search", search_node)
workflow.add_conditional_edges("call_llm", should_continue)
2. Gestión del estado en múltiples componentes
Cuando necesitas compartir y coordinar el estado entre diferentes partes de tu flujo de trabajo, la gestión explícita del estado de la API de Graph resulta beneficiosa.
# Multiple nodes can access and modify shared state
class WorkflowState(TypedDict):
    user_input: str
    search_results: list
    generated_response: str
    validation_status: str

def search_node(state):
    # Access shared state
    results = search(state["user_input"])
    return {"search_results": results}

def validation_node(state):
    # Access results from previous node
    is_valid = validate(state["generated_response"])
    return {"validation_status": "valid" if is_valid else "invalid"}
3. Procesamiento paralelo con sincronización
Cuando necesite ejecutar varias operaciones en paralelo y luego combinar sus resultados, la API de Graph lo gestiona de forma natural.
# Parallel processing of multiple data sources
workflow.add_node("fetch_news", fetch_news)
workflow.add_node("fetch_weather", fetch_weather)
workflow.add_node("fetch_stocks", fetch_stocks)
workflow.add_node("combine_data", combine_all_data)

# All fetch operations run in parallel
workflow.add_edge(START, "fetch_news")
workflow.add_edge(START, "fetch_weather")
workflow.add_edge(START, "fetch_stocks")

# Combine waits for all parallel operations to complete
workflow.add_edge("fetch_news", "combine_data")
workflow.add_edge("fetch_weather", "combine_data")
workflow.add_edge("fetch_stocks", "combine_data")
4. Desarrollo y documentación del equipo
La naturaleza visual de la API de Graph facilita a los equipos la comprensión, la documentación y el mantenimiento de flujos de trabajo complejos.
# Clear separation of concerns - each team member can work on different nodes
workflow.add_node("data_ingestion", data_team_function)
workflow.add_node("ml_processing", ml_team_function)
workflow.add_node("business_logic", product_team_function)
workflow.add_node("output_formatting", frontend_team_function)
​
Cuándo usar la API funcional
La API funcional utiliza un enfoque imperativo que integra las características de LangGraph en el código procedimental estándar.
1. Código de procedimiento existente
Cuando se dispone de código existente que utiliza un flujo de control estándar y se desea agregar características de LangGraph con una refactorización mínima.
# Functional API: Minimal changes to existing code
from langgraph.func import entrypoint, task

@task
def process_user_input(user_input: str) -> dict:
    # Existing function with minimal changes
    return {"processed": user_input.lower().strip()}

@entrypoint(checkpointer=checkpointer)
def workflow(user_input: str) -> str:
    # Standard Python control flow
    processed = process_user_input(user_input).result()

    if "urgent" in processed["processed"]:
        response = handle_urgent_request(processed).result()
    else:
        response = handle_normal_request(processed).result()

    return response
2. Flujos de trabajo lineales con lógica simple
Cuando tu flujo de trabajo es principalmente secuencial con una lógica condicional sencilla.
@entrypoint(checkpointer=checkpointer)
def essay_workflow(topic: str) -> dict:
    # Linear flow with simple branching
    outline = create_outline(topic).result()

    if len(outline["points"]) < 3:
        outline = expand_outline(outline).result()

    draft = write_draft(outline).result()

    # Human review checkpoint
    feedback = interrupt({"draft": draft, "action": "Please review"})

    if feedback == "approve":
        final_essay = draft
    else:
        final_essay = revise_essay(draft, feedback).result()

    return {"essay": final_essay}
3. Prototipado rápido
Cuando se desea probar ideas rápidamente sin la complejidad de definir esquemas de estado y estructuras gráficas.
@entrypoint(checkpointer=checkpointer)
def quick_prototype(data: dict) -> dict:
    # Fast iteration - no state schema needed
    step1_result = process_step1(data).result()
    step2_result = process_step2(step1_result).result()

    return {"final_result": step2_result}
4. Gestión de estados con ámbito funcional
Cuando el estado está naturalmente limitado a funciones individuales y no necesita ser compartido ampliamente.
@task
def analyze_document(document: str) -> dict:
    # Local state management within function
    sections = extract_sections(document)
    summaries = [summarize(section) for section in sections]
    key_points = extract_key_points(summaries)

    return {
        "sections": len(sections),
        "summaries": summaries,
        "key_points": key_points
    }

@entrypoint(checkpointer=checkpointer)
def document_processor(document: str) -> dict:
    analysis = analyze_document(document).result()
    # State is passed between functions as needed
    return generate_report(analysis).result()
​
Combinando ambas API
Puedes usar ambas API juntas en la misma aplicación. Esto resulta útil cuando diferentes partes de tu sistema tienen requisitos distintos.
from langgraph.graph import StateGraph
from langgraph.func import entrypoint

# Complex multi-agent coordination using Graph API
coordination_graph = StateGraph(CoordinationState)
coordination_graph.add_node("orchestrator", orchestrator_node)
coordination_graph.add_node("agent_a", agent_a_node)
coordination_graph.add_node("agent_b", agent_b_node)

# Simple data processing using Functional API
@entrypoint()
def data_processor(raw_data: dict) -> dict:
    cleaned = clean_data(raw_data).result()
    transformed = transform_data(cleaned).result()
    return transformed

# Use the functional API result in the graph
def orchestrator_node(state):
    processed_data = data_processor.invoke(state["raw_data"])
    return {"processed_data": processed_data}
​
Migración entre API
​
De la API funcional a la API gráfica
Cuando tu flujo de trabajo funcional se vuelve complejo, puedes migrar a la API de Graph:
# Before: Functional API
@entrypoint(checkpointer=checkpointer)
def complex_workflow(input_data: dict) -> dict:
    step1 = process_step1(input_data).result()

    if step1["needs_analysis"]:
        analysis = analyze_data(step1).result()
        if analysis["confidence"] > 0.8:
            result = high_confidence_path(analysis).result()
        else:
            result = low_confidence_path(analysis).result()
    else:
        result = simple_path(step1).result()

    return result

# After: Graph API
class WorkflowState(TypedDict):
    input_data: dict
    step1_result: dict
    analysis: dict
    final_result: dict

def should_analyze(state):
    return "analyze" if state["step1_result"]["needs_analysis"] else "simple_path"

def confidence_check(state):
    return "high_confidence" if state["analysis"]["confidence"] > 0.8 else "low_confidence"

workflow = StateGraph(WorkflowState)
workflow.add_node("step1", process_step1_node)
workflow.add_conditional_edges("step1", should_analyze)
workflow.add_node("analyze", analyze_data_node)
workflow.add_conditional_edges("analyze", confidence_check)
# ... add remaining nodes and edges
​
De grafo a API funcional
Cuando tu gráfico se vuelve demasiado complejo para procesos lineales simples:
# Before: Over-engineered Graph API
class SimpleState(TypedDict):
    input: str
    step1: str
    step2: str
    result: str

# After: Simplified Functional API
@entrypoint(checkpointer=checkpointer)
def simple_workflow(input_data: str) -> str:
    step1 = process_step1(input_data).result()
    step2 = process_step2(step1).result()
    return finalize_result(step2).result()
​
Resumen
Elija la API de Graph cuando necesite un control explícito sobre la estructura del flujo de trabajo, ramificaciones complejas, procesamiento paralelo o ventajas para la colaboración en equipo.
Elija la API funcional cuando desee agregar características de LangGraph al código existente con cambios mínimos, tenga flujos de trabajo lineales simples o necesite capacidades de creación rápida de prototipos.
Ambas API proporcionan las mismas características básicas de LangGraph (persistencia, transmisión de datos, intervención humana, memoria), pero las empaquetan en paradigmas diferentes para adaptarse a distintos estilos de desarrollo y casos de uso.

### Descripción general de la API de Graph

Gráficos
En esencia, LangGraph modela los flujos de trabajo de los agentes como grafos. El comportamiento de los agentes se define mediante tres componentes clave:
State: Una estructura de datos compartida que representa la instantánea actual de su aplicación. Puede ser de cualquier tipo de dato, pero normalmente se define mediante un esquema de estado compartido.
Nodes: Funciones que codifican la lógica de tus agentes. Reciben el estado actual como entrada, realizan algún cálculo o efecto secundario y devuelven un estado actualizado.
EdgesFunciones que determinan cuál Nodeejecutar a continuación en función del estado actual. Pueden ser bifurcaciones condicionales o transiciones fijas.
Al combinar Nodesy Edges, puedes crear flujos de trabajo complejos y cíclicos que hacen evolucionar el estado con el tiempo. Sin embargo, su verdadero potencial reside en cómo LangGraph gestiona ese estado.
Para enfatizar: Nodesy Edgesno son más que funciones; pueden contener un LLM o simplemente buen código de siempre.
En resumen: los nodos hacen el trabajo, las aristas indican qué hacer a continuación .
El algoritmo gráfico subyacente de LangGraph utiliza el paso de mensajes para definir un programa general. Cuando un nodo completa su operación, envía mensajes a través de una o más aristas a otros nodos. Estos nodos receptores ejecutan sus funciones, pasan los mensajes resultantes al siguiente conjunto de nodos y el proceso continúa. Inspirado en el sistema Pregel de Google , el programa se desarrolla en "superpasos" discretos.
Un superpaso puede considerarse una única iteración sobre los nodos del grafo. Los nodos que se ejecutan en paralelo forman parte del mismo superpaso, mientras que los que se ejecutan secuencialmente pertenecen a superpasos distintos. Al inicio de la ejecución del grafo, todos los nodos comienzan en un inactiveestado. Un nodo se convierte en activo activecuando recibe un nuevo mensaje (estado) en cualquiera de sus aristas entrantes (o "canales"). El nodo activo ejecuta entonces su función y responde con actualizaciones. Al final de cada superpaso, los nodos sin mensajes entrantes votan haltmarcándose como inactivos inactive. La ejecución del grafo finaliza cuando todos los nodos están inactivos inactivey no hay mensajes en tránsito.
​
Gráfico de estados
Esta StateGraphclase es la clase gráfica principal que se debe utilizar. Está parametrizada por un Stateobjeto definido por el usuario.
​
Compilando su gráfico
Para construir tu grafo, primero defines el estado , luego agregas nodos y aristas , y finalmente lo compilas. ¿Qué es exactamente compilar un grafo y por qué es necesario?
La compilación es un paso bastante sencillo. Realiza algunas comprobaciones básicas sobre la estructura del grafo (sin nodos huérfanos, etc.). También es donde se pueden especificar argumentos de tiempo de ejecución, como puntos de control y puntos de interrupción. Para compilar el grafo, basta con llamar al .compilemétodo:
graph = graph_builder.compile(...)
Debe compilar su gráfico antes de poder utilizarlo.
​
Estado
Lo primero que se hace al definir un grafo es definir su Stateesquema. Este esquema Stateconsta del esquema del grafo , así como reducerde funciones que especifican cómo aplicar las actualizaciones al estado. El esquema del esquema Stateserá el esquema de entrada para todos Nodeslos elementos Edgesdel grafo, y puede ser un modelo TypedDicto un Pydanticmodelo. Todos los Nodeselementos emitirán actualizaciones al esquema, Stateque luego se aplicarán mediante la función especificada reducer.
​
Esquema
La principal forma documentada de especificar el esquema de un grafo es mediante el uso de un TypedDict. Si desea proporcionar valores predeterminados en su estado, utilice un dataclass. También admitimos el uso de un Pydantic BaseModelcomo estado de su grafo si desea validación de datos recursiva (aunque tenga en cuenta que Pydantic tiene un rendimiento inferior al de un TypedDicto dataclass).
Por defecto, el gráfico tendrá los mismos esquemas de entrada y salida. Si desea cambiar esto, también puede especificar esquemas de entrada y salida explícitos directamente. Esto resulta útil cuando se tienen muchas claves, algunas de las cuales son de entrada y otras de salida. Consulte la guía para obtener más información.
create_agentLa fábrica de nivel superior langchainno admite esquemas de estado Pydantic.
​
Múltiples esquemas
Normalmente, todos los nodos del grafo se comunican con un único esquema. Esto significa que leerán y escribirán en los mismos canales de estado. Pero hay casos en los que queremos tener más control sobre esto:
Los nodos internos pueden transmitir información que no es necesaria en la entrada/salida del gráfico.
También podríamos querer usar diferentes esquemas de entrada/salida para el grafo. La salida podría, por ejemplo, contener solo una única clave de salida relevante.
Es posible que los nodos escriban en canales de estado privados dentro del grafo para la comunicación interna de los nodos. Podemos simplemente definir un esquema privado, PrivateState.
También es posible definir esquemas de entrada y salida explícitos para un grafo. En estos casos, definimos un esquema «interno» que contiene todas las claves relevantes para las operaciones del grafo. Además, definimos esquemas inputsecundarios outputque son subconjuntos del esquema «interno» para restringir la entrada y la salida del grafo. Consulte la sección « Definir esquemas de entrada y salida» para obtener más detalles.
Veamos un ejemplo:

```python
class InputState(TypedDict):
    user_input: str

class OutputState(TypedDict):
    graph_output: str

class OverallState(TypedDict):
    foo: str
    user_input: str
    graph_output: str

class PrivateState(TypedDict):
    bar: str

def node_1(state: InputState) -> OverallState:
    # Write to OverallState
    return {"foo": state["user_input"] + " name"}

def node_2(state: OverallState) -> PrivateState:
    # Read from OverallState, write to PrivateState
    return {"bar": state["foo"] + " is"}

def node_3(state: PrivateState) -> OutputState:
    # Read from PrivateState, write to OutputState
    return {"graph_output": state["bar"] + " Lance"}

builder = StateGraph(OverallState,input_schema=InputState,output_schema=OutputState)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)
builder.add_edge(START, "node_1")
builder.add_edge("node_1", "node_2")
builder.add_edge("node_2", "node_3")
builder.add_edge("node_3", END)

graph = builder.compile()
graph.invoke({"user_input":"My"})
# {'graph_output': 'My name is Lance'}
```

Hay dos puntos sutiles e importantes que cabe destacar:
Pasamos state: InputStatecomo esquema de entrada a node_1. Pero escribimos en foo, un canal en OverallState. ¿Cómo podemos escribir en un canal de estado que no está incluido en el esquema de entrada? Esto se debe a que un nodo puede escribir en cualquier canal de estado en el estado del grafo. El estado del grafo es la unión de los canales de estado definidos en la inicialización, que incluye OverallStatey los filtros InputStatey OutputState.
Inicializamos el gráfico con:

```python
StateGraph(
    OverallState,
    input_schema=InputState,
    output_schema=OutputState
)
```

¿Cómo podemos escribir PrivateStateen node_2? ¿Cómo accede el gráfico a este esquema si no se pasó en la StateGraphinicialización?
Podemos hacerlo porque _nodestambién podemos declarar estados adicionales channels_siempre que exista la definición del esquema de estado. En este caso, el PrivateStateesquema está definido, por lo que podemos agregarlo barcomo un nuevo canal de estado en el grafo y escribir en él.
​
Reductores
Los reductores son fundamentales para comprender cómo se aplican las actualizaciones de los nodos State. Cada clave Statetiene su propia función reductora independiente. Si no se especifica explícitamente ninguna función reductora, se asume que todas las actualizaciones de esa clave deben sobrescribirla. Existen varios tipos de reductores, comenzando con el tipo predeterminado:
​
reductor predeterminado
Estos dos ejemplos muestran cómo usar el reductor predeterminado:
Ejemplo A

```python
from typing_extensions import TypedDict

class State(TypedDict):
    foo: int
    bar: list[str]
```

En este ejemplo, no se especifican funciones reductoras para ninguna clave. Supongamos que la entrada al gráfico es:
{"foo": 1, "bar": ["hi"]}Supongamos entonces que el primero Nodedevuelve {"foo": 2}. Esto se trata como una actualización del estado. Nótese que Nodeno necesita devolver todo el Stateesquema, solo una actualización. Después de aplicar esta actualización, Statesería {"foo": 2, "bar": ["hi"]}. Si el segundo nodo devuelve , {"bar": ["bye"]}entonces Statesería{"foo": 2, "bar": ["bye"]}
Ejemplo B

```python
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: int
    bar: Annotated[list[str], add]
```

En este ejemplo, hemos usado el Annotatedtipo para especificar una función reductora ( operator.add) para la segunda clave ( bar). Nótese que la primera clave permanece sin cambios. Supongamos que la entrada al grafo es {"foo": 1, "bar": ["hi"]}. Supongamos entonces que el primero Nodedevuelve {"foo": 2}. Esto se trata como una actualización del estado. Nótese que Nodeno necesita devolver todo el Stateesquema, solo una actualización. Después de aplicar esta actualización, Statesería {"foo": 2, "bar": ["hi"]}. Si el segundo nodo devuelve , {"bar": ["bye"]}entonces Statesería {"foo": 2, "bar": ["hi", "bye"]}. Nótese aquí que la barclave se actualiza sumando las dos listas.
​
Exagerar
En algunos casos, es posible que desee omitir un reductor y sobrescribir directamente un valor de estado. LangGraph proporciona el Overwritetipo para este propósito. Aprenda cómo usarlo Overwriteaquí .
​
Trabajar con mensajes en el estado del gráfico
​
¿Por qué usar mensajes?
La mayoría de los proveedores modernos de LLM cuentan con una interfaz de chat que acepta una lista de mensajes como entrada. La interfaz de chat de LangChain , en particular, acepta una lista de objetos de mensaje como entrada. Estos mensajes se presentan en diversos formatos, como HumanMessage(entrada del usuario) o AIMessage(respuesta del LLM).
Para obtener más información sobre qué son los objetos de mensaje, consulte la guía conceptual de Mensajes .
​
Utilizar mensajes en tu gráfico
En muchos casos, es útil almacenar el historial de conversaciones anteriores como una lista de mensajes en el estado del grafo. Para ello, podemos agregar una clave (canal) al estado del grafo que almacene una lista de Messageobjetos y anotarla con una función reductora (ver messagesclave en el ejemplo a continuación). La función reductora es vital para indicarle al grafo cómo actualizar la lista de Messageobjetos en el estado con cada actualización de estado (por ejemplo, cuando un nodo envía una actualización). Si no se especifica una función reductora, cada actualización de estado sobrescribirá la lista de mensajes con el valor proporcionado más recientemente. Si se desea simplemente agregar mensajes a la lista existente, se puede usar operator.addcomo una función reductora.
Sin embargo, también es posible que desee actualizar manualmente los mensajes en el estado de su grafo (por ejemplo, mediante intervención humana). Si utilizara operator.add, las actualizaciones manuales del estado que envíe al grafo se agregarían a la lista existente de mensajes, en lugar de actualizar los mensajes existentes. Para evitar esto, necesita un reductor que pueda realizar un seguimiento de los ID de los mensajes y sobrescribir los mensajes existentes, si se actualizan. Para lograr esto, puede utilizar la add_messagesfunción predefinida. Para los mensajes nuevos, simplemente los agregará a la lista existente, pero también gestionará correctamente las actualizaciones de los mensajes existentes.
​
Publicación por entregas
Además de realizar un seguimiento de los identificadores de los mensajes, la add_messagesfunción también intentará deserializar los mensajes en Messageobjetos LangChain cada vez que se reciba una actualización de estado en el messagescanal.
Para obtener más información, consulte la serialización/deserialización de LangChain . Esto permite enviar entradas de grafos/actualizaciones de estado en el siguiente formato:
# this is supported
{"messages": [HumanMessage(content="message")]}

# and this is also supported
{"messages": [{"type": "human", "content": "message"}]}
Dado que las actualizaciones de estado siempre se deserializan en LangChain Messagescuando se utiliza add_messages, debe utilizar la notación de punto para acceder a los atributos del mensaje, como state["messages"][-1].content.
A continuación se muestra un ejemplo de un gráfico que utiliza add_messagescomo función reductora.

```python
from langchain.messages import AnyMessage
from langgraph.graph.message import add_messages
from typing import Annotated
from typing_extensions import TypedDict

class GraphState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
​```

Estado de los mensajes
Dado que tener una lista de mensajes en el estado es muy común, existe un estado predefinido MessagesStateque facilita el uso de mensajes. MessagesStateSe define con una sola messagesclave que es una lista de AnyMessageobjetos y utiliza el add_messagesreductor. Normalmente, hay más estado que gestionar además de los mensajes, por lo que vemos que se crean subclases de este estado y se añaden más campos, como:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    documents: list[str]
​```

Nodos
En LangGraph, los nodos son funciones de Python (síncronas o asíncronas) que aceptan los siguientes argumentos:
state—El estado del gráfico
config—Un RunnableConfigobjeto que contiene información de configuración como thread_ide información de seguimiento comotags
runtime—Un Runtimeobjeto que contiene información de tiempo de ejecucióncontext y otra información como store, stream_writer, execution_info, yserver_info
De forma similar a NetworkX, se añaden estos nodos a un gráfico utilizando el add_nodemétodo:

```python
from dataclasses import dataclass
from typing_extensions import TypedDict

from langgraph.graph import StateGraph
from langgraph.runtime import Runtime

class State(TypedDict):
    input: str
    results: str

@dataclass
class Context:
    user_id: str

builder = StateGraph(State)

def plain_node(state: State):
    return state

def node_with_runtime(state: State, runtime: Runtime[Context]):
    print("In node: ", runtime.context.user_id)
    return {"results": f"Hello, {state['input']}!"}

def node_with_execution_info(state: State, runtime: Runtime):
    print("In node with thread_id: ", runtime.execution_info.thread_id)
    return {"results": f"Hello, {state['input']}!"}


builder.add_node("plain_node", plain_node)
builder.add_node("node_with_runtime", node_with_runtime)
builder.add_node("node_with_execution_info", node_with_execution_info)
...
```

Detrás de escena, las funciones se convierten a RunnableLambda, lo que agrega soporte para procesamiento por lotes y asíncrono a su función, junto con el rastreo y la depuración nativos .
Si agrega un nodo a un gráfico sin especificar un nombre, se le asignará un nombre predeterminado equivalente al nombre de la función.
builder.add_node(my_node)
# You can then create edges to/from this node by referencing it as `"my_node"`
​
## STARTnodo
El STARTNodo es un nodo especial que representa el nodo que envía la entrada del usuario al grafo. El propósito principal de hacer referencia a este nodo es determinar qué nodos deben llamarse primero.

```python
from langgraph.graph import START

graph.add_edge(START, "node_a")
​
ENDnodo
El ENDnodo es un nodo especial que representa un nodo terminal. Se hace referencia a este nodo cuando se desea indicar qué aristas no tienen acciones después de haber sido ejecutadas.
from langgraph.graph import END

graph.add_edge("node_a", END)
​```

# Almacenamiento en caché de nodos
LangGraph admite el almacenamiento en caché de tareas/nodos en función de la entrada al nodo. Para usar el almacenamiento en caché:
Especifique una caché al compilar un gráfico (o al especificar un punto de entrada).
Especifique una política de caché para los nodos. Cada política de caché admite:
key_funcSe utiliza para generar una clave de caché basada en la entrada de un nodo, que por defecto es una hashde la entrada con pickle.
ttl, el tiempo de vida de la caché en segundos. Si no se especifica, la caché nunca caducará.
Por ejemplo:

```python
import time
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.cache.memory import InMemoryCache
from langgraph.types import CachePolicy


class State(TypedDict):
    x: int
    result: int


builder = StateGraph(State)

def expensive_node(state: State) -> dict[str, int]:
    # expensive computation
    time.sleep(2)
    return {"result": state["x"] * 2}

builder.add_node("expensive_node", expensive_node, cache_policy=CachePolicy(ttl=3))
builder.set_entry_point("expensive_node")
builder.set_finish_point("expensive_node")

graph = builder.compile(cache=InMemoryCache())

print(graph.invoke({"x": 5}, stream_mode='updates'))
# [{'expensive_node': {'result': 10}}]
print(graph.invoke({"x": 5}, stream_mode='updates'))
# [{'expensive_node': {'result': 10}, '__metadata__': {'cached': True}}]
```

La primera ejecución tarda dos segundos (debido a la simulación de cálculos costosos).
La segunda ejecución utiliza la caché y devuelve un resultado rápidamente.
​
Bordes
Las aristas definen cómo se enruta la lógica y cómo el grafo decide detenerse. Esto es fundamental para el funcionamiento de los agentes y la comunicación entre los distintos nodos. Existen varios tipos clave de aristas:
Aristas normales: Van directamente de un nodo al siguiente.
Aristas condicionales: Llama a una función para determinar a qué nodo(s) ir a continuación.
Punto de entrada: Qué nodo llamar primero cuando llega la entrada del usuario.
Punto de entrada condicional: Llama a una función para determinar qué nodo(s) llamar primero cuando llega la entrada del usuario.
Un nodo puede tener múltiples aristas salientes. Si un nodo tiene múltiples aristas salientes, todos esos nodos de destino se ejecutarán en paralelo como parte del siguiente superpaso.
​
Bordes normales
Si siempre quieres ir del nodo A al nodo B, puedes usar el add_edgemétodo directamente.
graph.add_edge("node_a", "node_b")
​
Aristas condicionales
Si desea enrutar opcionalmente a uno o más bordes (o finalizar opcionalmente), puede usar este add_conditional_edgesmétodo. Este método acepta el nombre de un nodo y una "función de enrutamiento" que se llamará después de que se ejecute ese nodo:
graph.add_conditional_edges("node_a", routing_function)
De forma similar a los nodos, routing_functionacepta la corriente statedel grafo y devuelve un valor.
Por defecto, el valor de retorno routing_functionse utiliza como nombre del nodo (o lista de nodos) al que se envía el estado para el siguiente paso. Todos esos nodos se ejecutarán en paralelo como parte del siguiente superpaso.
Opcionalmente, puede proporcionar un diccionario que asocie la routing_functionsalida de 's con el nombre del siguiente nodo.
graph.add_conditional_edges("node_a", routing_function, {True: "node_b", False: "node_c"})
Utilice Commanden lugar de aristas condicionales si desea combinar actualizaciones de estado y enrutamiento en una sola función.
​
Punto de entrada
El punto de entrada es el primer nodo o nodos que se ejecutan al iniciar el grafo. Puedes usar el add_edgemétodo del STARTnodo virtual al primer nodo que se ejecuta para especificar dónde entrar al grafo.
from langgraph.graph import START

graph.add_edge(START, "node_a")
​
Punto de entrada condicional
Un punto de entrada condicional permite comenzar en diferentes nodos según la lógica personalizada. Para ello, puede utilizar add_conditional_edgesel nodo virtual .START
from langgraph.graph import START

graph.add_conditional_edges(START, routing_function)
Opcionalmente, puede proporcionar un diccionario que asocie la routing_functionsalida de 's con el nombre del siguiente nodo.
graph.add_conditional_edges(START, routing_function, {True: "node_b", False: "node_c"})
​
Send
Por defecto, Nodesy Edgesse definen con antelación y operan sobre el mismo estado compartido. Sin embargo, puede haber casos en los que las aristas exactas no se conozcan de antemano y/o se desee que Stateexistan diferentes versiones de al mismo tiempo. Un ejemplo común de esto se encuentra en los patrones de diseño map-reduce . En este patrón de diseño, un primer nodo puede generar una lista de objetos, y se puede aplicar otro nodo a todos esos objetos. El número de objetos puede ser desconocido de antemano (lo que significa que el número de aristas puede ser desconocido) y la entrada Stateal flujo descendente Nodedebe ser diferente (una para cada objeto generado).
Para dar soporte a este patrón de diseño, LangGraph admite la devolución de Sendobjetos desde aristas condicionales. SendToma dos argumentos: primero es el nombre del nodo y segundo es el estado que se le pasará a ese nodo.

```python
def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state['subjects']]

graph.add_conditional_edges("node_a", continue_to_jokes)
​```

Command
CommandEs una primitiva versátil para controlar la ejecución de gráficos. Acepta cuatro parámetros:
update: Aplicar actualizaciones de estado (similar a devolver actualizaciones desde un nodo).
goto: Navegar a nodos específicos (similar a las aristas condicionales ).
graph: Seleccione un grafo padre al navegar desde subgrafos .
resume: Proporcione un valor para reanudar la ejecución después de una interrupción .
CommandSe utiliza en tres contextos:
Retorno desde los nodos : utiliceupdate,goto, ygraphpara combinar las actualizaciones de estado con el flujo de control.
Entrada a invoke/stream : Úseloresumepara continuar la ejecución después de una interrupción.
Retorno desde herramientas : Similar al retorno desde nodos, combina actualizaciones de estado y flujo de control desde dentro de una herramienta.
​
Regreso desde los nodos
​
Regresar Commandde las funciones del nodo para actualizar el estado y enrutar al siguiente nodo en un solo paso:

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

Con esto Commandtambién se puede lograr un comportamiento de flujo de control dinámico (idéntico a los bordes condicionales ):

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    if state["foo"] == "bar":
        return Command(update={"foo": "baz"}, goto="my_other_node")
```

Úselo Commandcuando necesite actualizar el estado y redirigir a un nodo diferente. Si solo necesita redirigir sin actualizar el estado, utilice aristas condicionales .
Al devolver Commanden sus funciones de nodo, debe agregar anotaciones de tipo de retorno con la lista de nombres de nodos a los que el nodo está enrutando, por ejemplo Command[Literal["my_other_node"]]. Esto es necesario para la representación del gráfico y le indica a LangGraph que my_nodepuede navegar a my_other_node.
Commandsolo agrega bordes dinámicos; los bordes estáticos definidos con add_edge/ addEdgeaún se ejecutan. Por ejemplo, si node_adevuelve Command(goto="my_other_node")y también tiene graph.add_edge("node_a", "node_b"), tanto node_bcomo my_other_nodese ejecutarán.
Consulta esta guía práctica para ver un ejemplo completo de cómo usarlo Command.
​
graph
Si está utilizando subgrafos , puede navegar desde un nodo dentro de un subgrafo a un nodo diferente en el grafo padre especificando graph=Command.PARENTen Command:

```python
def my_node(state: State) -> Command[Literal["other_subgraph"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
```

graphAl configurarlo , Command.PARENTse navegará al gráfico principal más cercano.
Cuando envías actualizaciones desde un nodo de subgrafo a un nodo de grafo padre para una clave que es compartida por los esquemas de estado del grafo padre y del subgrafo , debes definir un reductor para la clave que estás actualizando en el estado del grafo padre. Consulta este ejemplo .
Esto resulta especialmente útil al implementar traspasos entre múltiples agentes . Consulte Navegar a un nodo en un grafo principal para obtener más detalles.
​
Entrada a invoke/stream
Command(resume=...)es el único Command patrón destinado como entrada para invoke()/ stream(). No lo use Command(update=...)como entrada para continuar conversaciones de varios turnos, ya que pasar cualquier Commandcomo entrada reanuda desde el último punto de control (es decir, el último paso que se ejecutó, no __start__), el gráfico parecerá bloqueado si ya ha terminado. Para continuar una conversación en un hilo existente, pase un diccionario de entrada simple:

# WRONG - graph resumes from the latest checkpoint
# (last step that ran), appears stuck

```python
graph.invoke(Command(update={
    "messages": [{"role": "user", "content": "follow up"}]
}), config)

# CORRECT - plain dict restarts from __start__
graph.invoke( {
    "messages": [{"role": "user", "content": "follow up"}]
}, config)
​```

resume
Se utiliza Command(resume=...)para proporcionar un valor y reanudar la ejecución del gráfico después de una interrupción . El valor pasado resumese convierte en el valor de retorno de la interrupt()llamada dentro del nodo en pausa:

```python
from langgraph.types import Command, interrupt

def human_review(state: State):
    # Pauses the graph and waits for a value
    answer = interrupt("Do you approve?")
    return {"messages": [{"role": "user", "content": answer}]}

# First invocation - hits the interrupt and pauses
result = graph.invoke({"messages": [...]}, config)

# Resume with a value - the interrupt() call returns "yes"
result = graph.invoke(Command(resume="yes"), config)
```

Consulta la guía conceptual sobre interrupciones para obtener información detallada sobre los patrones de interrupción, incluidas las interrupciones múltiples y los bucles de validación.
​
Regresar de las herramientas
Puedes regresar Commandde las herramientas para actualizar el estado del gráfico y controlar el flujo. Úsalas updatepara modificar el estado (por ejemplo, guardar la información del cliente consultada durante una conversación) y gotopara redirigir a un nodo específico una vez que la herramienta finalice.
Cuando se utiliza dentro de herramientas, gotoagrega una arista dinámica; cualquier arista estática ya definida en el nodo que llamó a la herramienta seguirá ejecutándose.
Consulte la sección " Usar herramientas internas" para obtener más detalles.
​
Migraciones de grafos
LangGraph puede gestionar fácilmente las migraciones de definiciones de grafos (nodos, aristas y estado) incluso cuando se utiliza un puntero de control para realizar un seguimiento del estado.
Para los hilos que se encuentran al final del gráfico (es decir, que no se interrumpen), puede cambiar toda la topología del gráfico (es decir, todos los nodos y aristas, eliminar, agregar, renombrar, etc.).
Para los hilos que actualmente se encuentran interrumpidos, admitimos todos los cambios de topología, excepto el cambio de nombre o la eliminación de nodos (ya que ese hilo podría estar a punto de entrar en un nodo que ya no existe). Si esto supone un obstáculo, póngase en contacto con nosotros para que podamos priorizar una solución.
Para modificar el estado, tenemos compatibilidad total hacia atrás y hacia adelante para agregar y eliminar claves.
Las claves de estado que se renombran pierden su estado guardado en los hilos existentes.
Las claves de estado cuyos tipos cambian de forma incompatible podrían causar problemas en los hilos con estado anterior al cambio; si esto supone un obstáculo, póngase en contacto con nosotros para que podamos priorizar una solución.
​
Contexto de tiempo de ejecución
Al crear un grafo, puede especificar un context_schemacontexto de ejecución que se pasa a los nodos. Esto resulta útil para transmitir información a los nodos que no forma parte del estado del grafo. Por ejemplo, podría querer pasar dependencias como el nombre del modelo o una conexión a la base de datos.

```python
@dataclass
class ContextSchema:
    llm_provider: str = "openai"

graph = StateGraph(State, context_schema=ContextSchema)
```

A continuación, puedes pasar este contexto al gráfico utilizando el contextparámetro del invokemétodo.
graph.invoke(inputs, context={"llm_provider": "anthropic"})
A continuación, podrá acceder y utilizar este contexto dentro de un nodo o una arista condicional:

```python
from langgraph.runtime import Runtime

def node_a(state: State, runtime: Runtime[ContextSchema]):
    llm = get_llm(runtime.context.llm_provider)
    # ...
```

Consulte la sección "Agregar configuración de tiempo de ejecución" para obtener un desglose completo de la configuración.
​
Límite de recursión
El límite de recursión establece el número máximo de superpasos que el grafo puede ejecutar durante una sola ejecución. Una vez alcanzado el límite, LangGraph generará una excepción GraphRecursionError. A partir de la versión 1.0.6, el límite de recursión predeterminado se establece en 1000 pasos. El límite de recursión se puede establecer en cualquier grafo en tiempo de ejecución y se pasa a invoke/ streama través del diccionario de configuración. Es importante destacar que recursion_limites una clave independiente configy no debe pasarse dentro de la configurableclave como el resto de la configuración definida por el usuario. Vea el ejemplo a continuación:
graph.invoke(inputs, config={"recursion_limit": 5}, context={"llm": "anthropic"})
Lea el apartado "Límite de recursión" para obtener más información sobre cómo funciona el límite de recursión.
​
Acceso y manejo del contador de recursión
El contador de pasos actual es accesible desde config["metadata"]["langgraph_step"]cualquier nodo, lo que permite gestionar la recursión de forma proactiva antes de alcanzar el límite. Esto posibilita la implementación de estrategias de degradación gradual en la lógica del grafo.
​
Cómo funciona
El contador de pasos se almacena en config["metadata"]["langgraph_step"]. La comprobación del límite de recursión sigue la lógica: step > stopdonde stop = step + recursion_limit + 1. Cuando se supera el límite, LangGraph genera una excepción GraphRecursionError.
​
Acceder al contador de pasos actual
Puedes acceder al contador de pasos actual dentro de cualquier nodo para supervisar el progreso de la ejecución.

```python
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph

def my_node(state: dict, config: RunnableConfig) -> dict:
    current_step = config["metadata"]["langgraph_step"]
    print(f"Currently on step: {current_step}")
    return state
​```

Manejo proactivo de la recursión
LangGraph proporciona un RemainingStepsvalor gestionado que indica cuántos pasos quedan antes de alcanzar el límite de recursión. Esto permite una degradación gradual dentro del grafo.

```python
from typing import Annotated, Literal
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps  # Managed value - tracks steps until limit

def reasoning_node(state: State) -> dict:
    # RemainingSteps is automatically populated by LangGraph
    remaining = state["remaining_steps"]

    # Check if we're running low on steps
    if remaining <= 2:
        return {"messages": ["Approaching limit, wrapping up..."]}

    # Normal processing
    return {"messages": ["thinking..."]}

def route_decision(state: State) -> Literal["reasoning_node", "fallback_node"]:
    """Route based on remaining steps"""
    if state["remaining_steps"] <= 2:
        return "fallback_node"
    return "reasoning_node"

def fallback_node(state: State) -> dict:
    """Handle cases where recursion limit is approaching"""
    return {"messages": ["Reached complexity limit, providing best effort answer"]}

# Build graph
builder = StateGraph(State)
builder.add_node("reasoning_node", reasoning_node)
builder.add_node("fallback_node", fallback_node)
builder.add_edge(START, "reasoning_node")
builder.add_conditional_edges("reasoning_node", route_decision)
builder.add_edge("fallback_node", END)

graph = builder.compile()

# RemainingSteps works with any recursion_limit
result = graph.invoke({"messages": []}, {"recursion_limit": 10})
​```

Enfoques proactivos frente a reactivos
Existen dos enfoques principales para gestionar los límites de recursión: proactivo (monitorización dentro del grafo) y reactivo (captura de errores externamente).

```python
from typing import Annotated, Literal, TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.managed import RemainingSteps
from langgraph.errors import GraphRecursionError

class State(TypedDict):
    messages: Annotated[list, lambda x, y: x + y]
    remaining_steps: RemainingSteps

# Proactive Approach (recommended) - using RemainingSteps
def agent_with_monitoring(state: State) -> dict:
    """Proactively monitor and handle recursion within the graph"""
    remaining = state["remaining_steps"]

    # Early detection - route to internal handling
    if remaining <= 2:
        return {
            "messages": ["Approaching limit, returning partial result"]
        }

    # Normal processing
    return {"messages": [f"Processing... ({remaining} steps remaining)"]}

def route_decision(state: State) -> Literal["agent", END]:
    if state["remaining_steps"] <= 2:
        return END
    return "agent"

# Build graph
builder = StateGraph(State)
builder.add_node("agent", agent_with_monitoring)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", route_decision)
graph = builder.compile()

# Proactive: Graph completes gracefully
result = graph.invoke({"messages": []}, {"recursion_limit": 10})

# Reactive Approach (fallback) - catching error externally
try:
    result = graph.invoke({"messages": []}, {"recursion_limit": 10})
except GraphRecursionError as e:
    # Handle externally after graph execution fails
    result = {"messages": ["Fallback: recursion limit exceeded"]}
```

Las principales diferencias entre estos enfoques son:
Acercarse	Detección	Manejo	Flujo de control
Proactivo (usando RemainingSteps)	Antes de alcanzar el límite	Grafo interno mediante enrutamiento condicional	El gráfico continúa hasta el nodo de finalización.
Reactivo (captador GraphRecursionError)	Después de que se superó el límite	Gráfico externo en try/catch	Finalizada la ejecución del gráfico.
Ventajas proactivas:
Degradación gradual dentro del gráfico
Se puede guardar el estado intermedio en los puntos de control.
Mejor experiencia de usuario con resultados parciales
El gráfico se completa con normalidad (sin excepciones).
Ventajas reactivas:
Implementación más sencilla
No es necesario modificar la lógica del gráfico.
Gestión centralizada de errores
​
Otros metadatos disponibles
Además langgraph_step, también están disponibles los siguientes metadatos config["metadata"]:

```python
def inspect_metadata(state: dict, config: RunnableConfig) -> dict:
    metadata = config["metadata"]

    print(f"Step: {metadata['langgraph_step']}")
    print(f"Node: {metadata['langgraph_node']}")
    print(f"Triggers: {metadata['langgraph_triggers']}")
    print(f"Path: {metadata['langgraph_path']}")
    print(f"Checkpoint NS: {metadata['langgraph_checkpoint_ns']}")

    return state
​```

Visualización
A menudo resulta útil poder visualizar gráficos, sobre todo a medida que se vuelven más complejos. LangGraph incluye varias formas integradas de visualizar gráficos. Consulta la sección «Visualiza tu gráfico» para obtener más información.
​
Observabilidad y trazabilidad
Para rastrear, depurar y evaluar sus agentes, utilice LangSmith .

### Utilice la API de gráficos

Esta guía muestra los conceptos básicos de la API de grafos de LangGraph. Explica el estado y la composición de estructuras de grafos comunes, como secuencias , ramas y bucles . También abarca las funciones de control de LangGraph, incluyendo la API Send para flujos de trabajo map-reduce y la API Command para combinar actualizaciones de estado con saltos entre nodos.
​
Configuración
Instalar langgraph:

pip install -U langgraph
Configura LangSmith para una mejor depuración.
Regístrate en LangSmith para detectar rápidamente problemas y mejorar el rendimiento de tus proyectos LangGraph. LangSmith te permite usar datos de seguimiento para depurar, probar y supervisar tus aplicaciones LLM creadas con LangGraph. Consulta la documentación para obtener más información sobre cómo empezar .
​
Definir y actualizar el estado
Aquí mostramos cómo definir y actualizar el estado en LangGraph. Demostraremos lo siguiente:
Cómo usar el estado para definir el esquema de un grafo
Cómo utilizar reductores para controlar cómo se procesan las actualizaciones de estado.
​
Definir estado
El estado en LangGraph puede ser un TypedDictmodelo Pydantico una clase de datos. A continuación, utilizaremos un modelo TypedDict. Consulte la sección "Usar modelos Pydantic para el estado del grafo" para obtener más detalles sobre el uso de Pydantic.
Por defecto, los gráficos tendrán el mismo esquema de entrada y salida, y el estado determina dicho esquema. Consulte la sección «Definir esquemas de entrada y salida» para obtener información sobre cómo definir esquemas de entrada y salida distintos.
Consideremos un ejemplo sencillo con mensajes . Esto representa una formulación versátil del estado para muchas aplicaciones LLM. Consulte nuestra página de conceptos para obtener más detalles.

```python
from langchain.messages import AnyMessage
from typing_extensions import TypedDict

class State(TypedDict):
    messages: list[AnyMessage]
    extra_field: int
```
Este estado realiza un seguimiento de una lista de objetos de mensaje , así como de un campo entero adicional.
​
Estado de actualización
Vamos a crear un gráfico de ejemplo con un solo nodo. Nuestro nodo es simplemente una función de Python que lee el estado de nuestro gráfico y lo actualiza. El primer argumento de esta función siempre será el estado:

```python
from langchain.messages import AIMessage

def node(state: State):
    messages = state["messages"]
    new_message = AIMessage("Hello!")
    return {"messages": messages + [new_message], "extra_field": 10}
```

Este nodo simplemente agrega un mensaje a nuestra lista de mensajes y completa un campo adicional.
Los nodos deben devolver actualizaciones del estado directamente, en lugar de modificar el estado.
A continuación, definamos un grafo simple que contenga este nodo. Usamos StateGraphpara definir un grafo que opere sobre este estado. Luego usamos add_nodepara poblar nuestro grafo.

```python
from langgraph.graph import StateGraph

builder = StateGraph(State)
builder.add_node(node)
builder.set_entry_point("node")
graph = builder.compile()
```

LangGraph proporciona utilidades integradas para visualizar su gráfico. Examinemos nuestro gráfico. Consulte Visualice su gráfico para obtener más detalles sobre la visualización.

```python
from IPython.display import Image, display
display(Image(graph.get_graph().draw_mermaid_png()))
```

Grafo simple con un solo nodo
En este caso, nuestro gráfico solo ejecuta un único nodo. Procedamos con una invocación simple:

```python
from langchain.messages import HumanMessage

result = graph.invoke({"messages": [HumanMessage("Hi")]})
result
{'messages': [HumanMessage(content='Hi'), AIMessage(content='Hello!')], 'extra_field': 10}
```

Tenga en cuenta que:
Iniciamos la invocación actualizando una única clave del estado.
Recibimos el estado completo en el resultado de la invocación.
Para mayor comodidad, frecuentemente inspeccionamos el contenido de los objetos de mensaje mediante formato legible:

```python
for message in result["messages"]:
    message.pretty_print()
================================ Human Message ================================

Hi
================================== Ai Message ==================================

Hello!
​```

Actualizaciones de estado de procesos con reductores
Cada clave en el estado puede tener su propia función reductora independiente , que controla cómo se aplican las actualizaciones de los nodos. Si no se especifica explícitamente ninguna función reductora, se asume que todas las actualizaciones de la clave deben sobrescribirla.
Para TypedDictlos esquemas de estado, podemos definir reductores anotando el campo correspondiente del estado con una función reductora.
En el ejemplo anterior, nuestro nodo actualizó la "messages"clave en el estado agregándole un mensaje. A continuación, agregamos un reductor a esta clave, de modo que las actualizaciones se agreguen automáticamente:

```python
from typing_extensions import Annotated

def add(left, right):
    """Can also import `add` from the `operator` built-in."""
    return left + right

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add]
    extra_field: int
Ahora nuestro nodo se puede simplificar:
def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}
from langgraph.graph import START

graph = StateGraph(State).add_node(node).add_edge(START, "node").compile()

result = graph.invoke({"messages": [HumanMessage("Hi")]})

for message in result["messages"]:
    message.pretty_print()
================================ Human Message ================================

Hi
================================== Ai Message ==================================

Hello!
​```

# Estado de los mensajes
En la práctica, existen consideraciones adicionales para actualizar las listas de mensajes:
Es posible que deseemos actualizar un mensaje existente en el estado.
Es posible que deseemos aceptar abreviaturas para los formatos de mensajes , como el formato OpenAI .
LangGraph incluye un reductor incorporado add_messagesque maneja estas consideraciones:

```python
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    extra_field: int

def node(state: State):
    new_message = AIMessage("Hello!")
    return {"messages": [new_message], "extra_field": 10}

graph = StateGraph(State).add_node(node).set_entry_point("node").compile()
input_message = {"role": "user", "content": "Hi"}

result = graph.invoke({"messages": [input_message]})

for message in result["messages"]:
    message.pretty_print()
================================ Human Message ================================

Hi
================================== Ai Message ==================================
```

Hello!
Esta es una representación versátil del estado para aplicaciones que involucran modelos de chat . LangGraph incluye una versión predefinida MessagesStatepara mayor comodidad, de modo que podamos tener:

```python
from langgraph.graph import MessagesState

class State(MessagesState):
    extra_field: int
​```

Reductores de derivación conOverwrite
En algunos casos, es posible que desee omitir un reductor y sobrescribir directamente un valor de estado. LangGraph proporciona el Overwritetipo para este propósito. Cuando un nodo devuelve un valor envuelto con Overwrite, se omite el reductor y el canal se establece directamente a ese valor.
Esto resulta útil cuando se desea restablecer o reemplazar el estado acumulado en lugar de fusionarlo con los valores existentes.

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Overwrite
from typing_extensions import Annotated, TypedDict
import operator

class State(TypedDict):
    messages: Annotated[list, operator.add]

def add_message(state: State):
    return {"messages": ["first message"]}

def replace_messages(state: State):
    # Bypass the reducer and replace the entire messages list
    return {"messages": Overwrite(["replacement message"])}

builder = StateGraph(State)
builder.add_node("add_message", add_message)
builder.add_node("replace_messages", replace_messages)
builder.add_edge(START, "add_message")
builder.add_edge("add_message", "replace_messages")
builder.add_edge("replace_messages", END)

graph = builder.compile()

result = graph.invoke({"messages": ["initial"]})
print(result["messages"])
['replacement message']
También puede utilizar el formato JSON con la clave especial "__overwrite__":
def replace_messages(state: State):
    return {"messages": {"__overwrite__": ["replacement message"]}}
```

Cuando los nodos se ejecutan en paralelo, solo un nodo puede usar Overwritela misma clave de estado en un superpaso determinado. Si varios nodos intentan sobrescribir la misma clave en el mismo superpaso, se InvalidUpdateErrorgenerará una excepción.
​
Definir esquemas de entrada y salida
Por defecto, StateGraphfunciona con un único esquema, y ​​se espera que todos los nodos se comuniquen utilizando dicho esquema. Sin embargo, también es posible definir esquemas de entrada y salida distintos para un grafo.
Cuando se especifican esquemas distintos, se seguirá utilizando un esquema interno para la comunicación entre nodos. El esquema de entrada garantiza que la entrada proporcionada coincida con la estructura esperada, mientras que el esquema de salida filtra los datos internos para devolver solo la información relevante según el esquema de salida definido.
A continuación, veremos cómo definir esquemas de entrada y salida distintos.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# Define the schema for the input
class InputState(TypedDict):
    question: str

# Define the schema for the output
class OutputState(TypedDict):
    answer: str

# Define the overall schema, combining both input and output
class OverallState(InputState, OutputState):
    pass

# Define the node that processes the input and generates an answer
def answer_node(state: InputState):
    # Example answer and an extra key
    return {"answer": "bye", "question": state["question"]}

# Build the graph with input and output schemas specified
builder = StateGraph(OverallState, input_schema=InputState, output_schema=OutputState)
builder.add_node(answer_node)  # Add the answer node
builder.add_edge(START, "answer_node")  # Define the starting edge
builder.add_edge("answer_node", END)  # Define the ending edge
graph = builder.compile()  # Compile the graph

# Invoke the graph with an input and print the result
print(graph.invoke({"question": "hi"}))
{'answer': 'bye'}
```

Observe que la salida de invoke solo incluye el esquema de salida.
​
Pasar estado privado entre nodos
En algunos casos, es posible que desee que los nodos intercambien información crucial para la lógica intermedia, pero que no necesita formar parte del esquema principal del grafo. Estos datos privados no son relevantes para la entrada/salida general del grafo y solo deben compartirse entre ciertos nodos.
A continuación, crearemos un ejemplo de grafo secuencial que consta de tres nodos (nodo_1, nodo_2 y nodo_3), donde los datos privados se pasan entre los dos primeros pasos (nodo_1 y nodo_2), mientras que el tercer paso (nodo_3) solo tiene acceso al estado general público.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

# The overall state of the graph (this is the public state shared across nodes)
class OverallState(TypedDict):
    a: str

# Output from node_1 contains private data that is not part of the overall state
class Node1Output(TypedDict):
    private_data: str

# The private data is only shared between node_1 and node_2
def node_1(state: OverallState) -> Node1Output:
    output = {"private_data": "set by node_1"}
    print(f"Entered node `node_1`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Node 2 input only requests the private data available after node_1
class Node2Input(TypedDict):
    private_data: str

def node_2(state: Node2Input) -> OverallState:
    output = {"a": "set by node_2"}
    print(f"Entered node `node_2`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Node 3 only has access to the overall state (no access to private data from node_1)
def node_3(state: OverallState) -> OverallState:
    output = {"a": "set by node_3"}
    print(f"Entered node `node_3`:\n\tInput: {state}.\n\tReturned: {output}")
    return output

# Connect nodes in a sequence
# node_2 accepts private data from node_1, whereas
# node_3 does not see the private data.
builder = StateGraph(OverallState).add_sequence([node_1, node_2, node_3])
builder.add_edge(START, "node_1")
graph = builder.compile()

# Invoke the graph with the initial state
response = graph.invoke(
    {
        "a": "set at start",
    }
)

print()
print(f"Output of graph invocation: {response}")
Entered node `node_1`:
    Input: {'a': 'set at start'}.
    Returned: {'private_data': 'set by node_1'}
Entered node `node_2`:
    Input: {'private_data': 'set by node_1'}.
    Returned: {'a': 'set by node_2'}
Entered node `node_3`:
    Input: {'a': 'set by node_2'}.
    Returned: {'a': 'set by node_3'}

Output of graph invocation: {'a': 'set by node_3'}
​```

Utilice modelos pydantic para el estado del grafo
Un StateGraph acepta un state_schemaargumento en la inicialización que especifica la "forma" del estado al que los nodos del grafo pueden acceder y actualizar.
En nuestros ejemplos, normalmente usamos un TypedDicto nativo de Python dataclasspara state_schema, pero state_schemapuede ser de cualquier tipo .
Aquí veremos cómo se puede utilizar un Pydantic BaseModelstate_schema para agregar validación en tiempo de ejecución a las entradas .
Limitaciones conocidas
Actualmente, el resultado del gráfico NO será una instancia de un modelo pydantic.
La validación en tiempo de ejecución solo se produce en las entradas del primer nodo del gráfico, no en los nodos o salidas subsiguientes.
El registro de errores de validación de pydantic no muestra en qué nodo se produce el error.
La validación recursiva de Pydantic puede ser lenta. Para aplicaciones donde el rendimiento es crucial, es posible que desee considerar el uso de otra dataclassalternativa.

```python
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from pydantic import BaseModel

# The overall state of the graph (this is the public state shared across nodes)
class OverallState(BaseModel):
    a: str

def node(state: OverallState):
    return {"a": "goodbye"}

# Build the state graph
builder = StateGraph(OverallState)
builder.add_node(node)  # node_1 is the first node
builder.add_edge(START, "node")  # Start the graph with node_1
builder.add_edge("node", END)  # End the graph after node_1
graph = builder.compile()

# Test the graph with a valid input
graph.invoke({"a": "hello"})
Invocar el gráfico con una entrada no válida
try:
    graph.invoke({"a": 123})  # Should be a string
except Exception as e:
    print("An exception was raised because `a` is an integer rather than a string.")
    print(e)
An exception was raised because `a` is an integer rather than a string.
1 validation error for OverallState
a
  Input should be a valid string [type=string_type, input_value=123, input_type=int]
    For further information visit https://errors.pydantic.dev/2.9/v/string_type
```

Consulte a continuación las características adicionales del estado del modelo Pydantic:
Comportamiento de serialización

Al utilizar modelos Pydantic como esquemas de estado, es importante comprender cómo funciona la serialización, especialmente cuando:
Pasar objetos Pydantic como entradas
Recepción de resultados del gráfico
Trabajando con modelos Pydantic anidados
Veamos estos comportamientos en acción.

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class NestedModel(BaseModel):
    value: str

class ComplexState(BaseModel):
    text: str
    count: int
    nested: NestedModel

def process_node(state: ComplexState):
    # Node receives a validated Pydantic object
    print(f"Input state type: {type(state)}")
    print(f"Nested type: {type(state.nested)}")
    # Return a dictionary update
    return {"text": state.text + " processed", "count": state.count + 1}

# Build the graph
builder = StateGraph(ComplexState)
builder.add_node("process", process_node)
builder.add_edge(START, "process")
builder.add_edge("process", END)
graph = builder.compile()

# Create a Pydantic instance for input
input_state = ComplexState(text="hello", count=0, nested=NestedModel(value="test"))
print(f"Input object type: {type(input_state)}")

# Invoke graph with a Pydantic instance
result = graph.invoke(input_state)
print(f"Output type: {type(result)}")
print(f"Output content: {result}")

# Convert back to Pydantic model if needed
output_model = ComplexState(**result)
print(f"Converted back to Pydantic: {type(output_model)}")
Coerción de tipo en tiempo de ejecución

Pydantic realiza una conversión de tipos en tiempo de ejecución para ciertos tipos de datos. Esto puede ser útil, pero también puede provocar un comportamiento inesperado si no se tiene en cuenta.
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

class CoercionExample(BaseModel):
    # Pydantic will coerce string numbers to integers
    number: int
    # Pydantic will parse string booleans to bool
    flag: bool

def inspect_node(state: CoercionExample):
    print(f"number: {state.number} (type: {type(state.number)})")
    print(f"flag: {state.flag} (type: {type(state.flag)})")
    return {}

builder = StateGraph(CoercionExample)
builder.add_node("inspect", inspect_node)
builder.add_edge(START, "inspect")
builder.add_edge("inspect", END)
graph = builder.compile()

# Demonstrate coercion with string inputs that will be converted
result = graph.invoke({"number": "42", "flag": "true"})

# This would fail with a validation error
try:
    graph.invoke({"number": "not-a-number", "flag": "true"})
except Exception as e:
    print(f"\nExpected validation error: {e}")
```

Trabajar con modelos de mensajes

Al trabajar con tipos de mensajes LangChain en su esquema de estado, hay consideraciones importantes para la serialización. Debe usar AnyMessage(en lugar de BaseMessage) para una serialización/deserialización adecuada cuando utilice objetos de mensaje a través de la red.

```python
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from langchain.messages import HumanMessage, AIMessage, AnyMessage
from typing import List

class ChatState(BaseModel):
    messages: List[AnyMessage]
    context: str

def add_message(state: ChatState):
    return {"messages": state.messages + [AIMessage(content="Hello there!")]}

builder = StateGraph(ChatState)
builder.add_node("add_message", add_message)
builder.add_edge(START, "add_message")
builder.add_edge("add_message", END)
graph = builder.compile()

# Create input with a message
initial_state = ChatState(
    messages=[HumanMessage(content="Hi")], context="Customer support chat"
)

result = graph.invoke(initial_state)
print(f"Output: {result}")

# Convert back to Pydantic model to see message types
output_model = ChatState(**result)
for i, msg in enumerate(output_model.messages):
    print(f"Message {i}: {type(msg).__name__} - {msg.content}")
​```

Agregar configuración de tiempo de ejecución
A veces, querrás poder configurar tu gráfico al llamarlo. Por ejemplo, es posible que quieras poder especificar qué LLM o indicador del sistema usar en tiempo de ejecución, sin contaminar el estado del gráfico con estos parámetros .
Para agregar configuración en tiempo de ejecución:
Especifique un esquema para su configuración.
Agregue la configuración a la firma de la función para nodos o aristas condicionales.
Pasa la configuración al gráfico.
A continuación se muestra un ejemplo sencillo:

```python
from langgraph.graph import END, StateGraph, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

# 1. Specify config schema
class ContextSchema(TypedDict):
    my_runtime_value: str

# 2. Define a graph that accesses the config in a node
class State(TypedDict):
    my_state_value: str

def node(state: State, runtime: Runtime[ContextSchema]):
    if runtime.context["my_runtime_value"] == "a":
        return {"my_state_value": 1}
    elif runtime.context["my_runtime_value"] == "b":
        return {"my_state_value": 2}
    else:
        raise ValueError("Unknown values.")

builder = StateGraph(State, context_schema=ContextSchema)
builder.add_node(node)
builder.add_edge(START, "node")
builder.add_edge("node", END)

graph = builder.compile()

# 3. Pass in configuration at runtime:
print(graph.invoke({}, context={"my_runtime_value": "a"}))
print(graph.invoke({}, context={"my_runtime_value": "b"}))
{'my_state_value': 1}
{'my_state_value': 2}
```

Ejemplo ampliado: especificación de LLM en tiempo de ejecución

A continuación, mostramos un ejemplo práctico en el que configuramos qué modelo LLM utilizar en tiempo de ejecución. Usaremos modelos de OpenAI y Anthropic.

```python
from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, END, StateGraph, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

@dataclass
class ContextSchema:
    model_provider: str = "anthropic"

MODELS = {
    "anthropic": init_chat_model("claude-haiku-4-5-20251001"),
    "openai": init_chat_model("gpt-4.1-mini"),
}

def call_model(state: MessagesState, runtime: Runtime[ContextSchema]):
    model = MODELS[runtime.context.model_provider]
    response = model.invoke(state["messages"])
    return {"messages": [response]}

builder = StateGraph(MessagesState, context_schema=ContextSchema)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

# Usage
input_message = {"role": "user", "content": "hi"}
# With no configuration, uses default (Anthropic)
response_1 = graph.invoke({"messages": [input_message]}, context=ContextSchema())["messages"][-1]
# Or, can set OpenAI
response_2 = graph.invoke({"messages": [input_message]}, context={"model_provider": "openai"})["messages"][-1]

print(response_1.response_metadata["model_name"])
print(response_2.response_metadata["model_name"])
claude-haiku-4-5-20251001
gpt-4.1-mini-2025-04-14
Ejemplo ampliado: especificación del modelo y del mensaje del sistema en tiempo de ejecución.

A continuación, mostramos un ejemplo práctico en el que configuramos dos parámetros: el LLM y el mensaje del sistema que se utilizarán en tiempo de ejecución.
from dataclasses import dataclass
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

@dataclass
class ContextSchema:
    model_provider: str = "anthropic"
    system_message: str | None = None

MODELS = {
    "anthropic": init_chat_model("claude-haiku-4-5-20251001"),
    "openai": init_chat_model("gpt-4.1-mini"),
}

def call_model(state: MessagesState, runtime: Runtime[ContextSchema]):
    model = MODELS[runtime.context.model_provider]
    messages = state["messages"]
    if (system_message := runtime.context.system_message):
        messages = [SystemMessage(system_message)] + messages
    response = model.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState, context_schema=ContextSchema)
builder.add_node("model", call_model)
builder.add_edge(START, "model")
builder.add_edge("model", END)

graph = builder.compile()

# Usage
input_message = {"role": "user", "content": "hi"}
response = graph.invoke({"messages": [input_message]}, context={"model_provider": "openai", "system_message": "Respond in Italian."})
for message in response["messages"]:
    message.pretty_print()
================================ Human Message ================================

hi
================================== Ai Message ==================================
```

Ciao! Come posso aiutarti oggi?
​
Agregar políticas de reintento
Existen muchos casos de uso en los que puede que desee que su nodo tenga una política de reintentos personalizada, por ejemplo, si está llamando a una API, consultando una base de datos o llamando a un LLM, etc. LangGraph le permite agregar políticas de reintentos a los nodos.
Para configurar una política de reintentos, pase el retry_policyparámetro a add_node. El retry_policyparámetro acepta un RetryPolicyobjeto de tupla con nombre. A continuación, instanciamos un RetryPolicyobjeto con los parámetros predeterminados y lo asociamos con un nodo:

```python
from langgraph.types import RetryPolicy

builder.add_node(
    "node_name",
    node_function,
    retry_policy=RetryPolicy(),
)
```

Por defecto, el retry_onparámetro utiliza la default_retry_onfunción, que reintenta ante cualquier excepción, excepto en los siguientes casos:
ValueError
TypeError
ArithmeticError
ImportError
LookupError
NameError
SyntaxError
RuntimeError
ReferenceError
StopIteration
StopAsyncIteration
OSError
Además, para excepciones de bibliotecas de solicitudes HTTP populares como requestsy httpxsolo vuelve a intentarlo en códigos de estado 5xx.
Ejemplo ampliado: personalización de las políticas de reintento

Consideremos un ejemplo en el que leemos datos de una base de datos SQL. A continuación, pasamos dos políticas de reintento diferentes a los nodos:

```python
import sqlite3
from typing_extensions import TypedDict
from langchain.chat_models import init_chat_model
from langgraph.graph import END, MessagesState, StateGraph, START
from langgraph.types import RetryPolicy
from langchain_community.utilities import SQLDatabase
from langchain.messages import AIMessage

db = SQLDatabase.from_uri("sqlite:///:memory:")
model = init_chat_model("claude-haiku-4-5-20251001")

def query_database(state: MessagesState):
    query_result = db.run("SELECT * FROM Artist LIMIT 10;")
    return {"messages": [AIMessage(content=query_result)]}

def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": [response]}

# Define a new graph
builder = StateGraph(MessagesState)
builder.add_node(
    "query_database",
    query_database,
    retry_policy=RetryPolicy(retry_on=sqlite3.OperationalError),
)
builder.add_node("model", call_model, retry_policy=RetryPolicy(max_attempts=5))
builder.add_edge(START, "model")
builder.add_edge("model", "query_database")
builder.add_edge("query_database", END)
graph = builder.compile()
​```

Acceder a la información de ejecución dentro de un nodo
Puedes acceder a la identidad de ejecución y a la información de reintento a través de runtime.execution_info. Esto muestra los identificadores de subproceso, ejecución y punto de control, así como el estado de reintento, sin necesidad de leer configdirectamente desde .
Atributo	Tipo	Descripción
thread_id	str | None	ID del hilo para la ejecución actual. NoneSin puntero de control.
run_id	str | None	ID de ejecución para la ejecución actual. NoneCuando no se proporciona en la configuración.
checkpoint_id	str	ID del punto de control para la ejecución actual.
checkpoint_ns	str	Espacio de nombres del punto de control para la ejecución actual.
task_id	str	ID de la tarea para la ejecución actual.
node_attempt	int	Número de intento de ejecución actual (indexado desde 1). 1En el primer intento, 2en el primer reintento, etc.
node_first_attempt_time	float | None	Marca de tiempo Unix (en segundos) del momento en que comenzó el primer intento. Permanece igual en los reintentos.
​
IDs de subprocesos y ejecuciones
Se utiliza execution_infopara acceder al ID del hilo, al ID de ejecución y a otros campos de identidad dentro de un nodo:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

def my_node(state: State, runtime: Runtime):
    info = runtime.execution_info
    print(f"Thread: {info.thread_id}, Run: {info.run_id}")
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()
​```

Ajustar el comportamiento en función del estado de reintento
Cuando un nodo tiene una política de reintentos, úselo execution_infopara inspeccionar el número de intento actual y cambiar a una alternativa después de que falle el primer intento:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from langgraph.types import RetryPolicy
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

def my_node(state: State, runtime: Runtime):
    info = runtime.execution_info
    if info.node_attempt > 1:
        # use a fallback on retries
        return {"result": call_fallback_api()}
    return {"result": call_primary_api()}

builder = StateGraph(State)
builder.add_node("my_node", my_node, retry_policy=RetryPolicy(max_attempts=3))
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()
```

execution_infoestá disponible en el Runtimeobjeto incluso sin una política de reintentos; node_attemptpor defecto es 1y node_first_attempt_timese establece en el momento en que el nodo comienza a ejecutarse.
​
Acceder a la información del servidor dentro de un nodo
Cuando tu gráfico se ejecuta en LangGraph Server, puedes acceder a los metadatos específicos del servidor a través de runtime.server_info. Esto muestra el ID del asistente, el ID del gráfico y el usuario autenticado sin necesidad de leer directamente los metadatos de configuración o las claves configurables.
Atributo	Tipo	Descripción
assistant_id	str	El ID del asistente para la implementación actual.
graph_id	str	El ID del gráfico para la implementación actual.
user	BaseUser | None	El usuario autenticado, si se ha configurado la autenticación personalizada .

```python
from langgraph.graph import StateGraph, START, END
from langgraph.runtime import Runtime
from typing_extensions import TypedDict

class State(TypedDict):
    result: str

def my_node(state: State, runtime: Runtime):
    server = runtime.server_info
    if server is not None:
        print(f"Assistant: {server.assistant_id}, Graph: {server.graph_id}")
        if server.user is not None:
            print(f"User: {server.user.identity}")
    return {"result": "done"}

builder = StateGraph(State)
builder.add_node("my_node", my_node)
builder.add_edge(START, "my_node")
builder.add_edge("my_node", END)
graph = builder.compile()
```

server_infoEsto ocurre Nonecuando el gráfico no se está ejecutando en LangGraph Server (por ejemplo, durante el desarrollo o las pruebas locales).
Requiere deepagents>=0.5.0(o langgraph>=1.1.5) para runtime.execution_infoy runtime.server_info.
​
Agregar almacenamiento en caché de nodos
El almacenamiento en caché de nodos es útil cuando se desea evitar operaciones repetitivas, como al realizar tareas costosas (ya sea en términos de tiempo o costo). LangGraph permite agregar políticas de almacenamiento en caché individualizadas a los nodos de un grafo.
Para configurar una política de caché, pase el cache_policyparámetro a la add_nodefunción. En el siguiente ejemplo, CachePolicyse instancia un objeto con un tiempo de vida de 120 segundos y el key_funcgenerador predeterminado. Luego se asocia con un nodo:

```python
from langgraph.types import CachePolicy

builder.add_node(
    "node_name",
    node_function,
    cache_policy=CachePolicy(ttl=120),
)
```

Luego, para habilitar el almacenamiento en caché a nivel de nodo para un grafo, configure el cacheargumento al compilar el grafo. El siguiente ejemplo utiliza InMemoryCachepara configurar un grafo con caché en memoria, pero SqliteCachetambién está disponible.

```python
from langgraph.cache.memory import InMemoryCache

graph = builder.compile(cache=InMemoryCache())
​```

Crea una secuencia de pasos
Requisitos previos Esta guía presupone familiaridad con la sección anterior sobre el estado .
Aquí demostramos cómo construir una secuencia simple de pasos. Mostraremos:
Cómo construir un grafo secuencial
Sistema abreviado integrado para construir gráficos similares.
Para agregar una secuencia de nodos, utilizamos los métodos add_nodey add_edgede nuestro grafo :

```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

# Add nodes
builder.add_node(step_1)
builder.add_node(step_2)
builder.add_node(step_3)

# Add edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
También podemos utilizar la notación abreviada incorporada .add_sequence:
builder = StateGraph(State).add_sequence([step_1, step_2, step_3])
builder.add_edge(START, "step_1")
```

¿Por qué dividir los pasos de la aplicación en una secuencia con LangGraph?

LangGraph facilita la adición de una capa de persistencia subyacente a su aplicación. Esto permite que el estado se guarde en puntos de control entre la ejecución de los nodos, de modo que sus nodos LangGraph gobiernen:
Cómo se realizan las actualizaciones de estado
Cómo se reanudan las interrupciones en los flujos de trabajo con intervención humana.
Cómo podemos “rebobinar” y ramificar las ejecuciones utilizando las funciones de viaje en el tiempo de LangGraph.
También determinan cómo se transmiten los pasos de ejecución y cómo se visualiza y depura la aplicación mediante Studio .
Vamos a demostrar un ejemplo completo. Crearemos una secuencia de tres pasos:
Rellene un valor en una clave del estado
Actualizar el mismo valor
Introduce un valor diferente
Primero definamos nuestro estado . Este rige el esquema del grafo y también puede especificar cómo aplicar las actualizaciones. Consulte el apartado «Procesar actualizaciones de estado con reductores» para obtener más detalles.
En nuestro caso, solo haremos un seguimiento de dos valores:

```python
from typing_extensions import TypedDict

class State(TypedDict):
    value_1: str
    value_2: int
Nuestros nodos son simplemente funciones de Python que leen el estado de nuestro grafo y lo actualizan. El primer argumento de esta función siempre será el estado:
def step_1(state: State):
    return {"value_1": "a"}

def step_2(state: State):
    current_value_1 = state["value_1"]
    return {"value_1": f"{current_value_1} b"}

def step_3(state: State):
    return {"value_2": 10}

```

Tenga en cuenta que, al actualizar el estado, cada nodo puede especificar simplemente el valor de la clave que desea actualizar.
Por defecto, esto sobrescribirá el valor de la clave correspondiente. También puedes usar reductores para controlar cómo se procesan las actualizaciones; por ejemplo, puedes añadir actualizaciones sucesivas a una clave. Consulta Procesar actualizaciones de estado con reductores para obtener más información.
Finalmente, definimos el grafo. Usamos StateGraph para definir un grafo que opera sobre este estado.
Luego utilizaremos add_nodey add_edgepara completar nuestro gráfico y definir su flujo de control.

```python
from langgraph.graph import START, StateGraph

builder = StateGraph(State)

# Add nodes
builder.add_node(step_1)
builder.add_node(step_2)
builder.add_node(step_3)

# Add edges
builder.add_edge(START, "step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
```

Especificar nombres personalizados Puede especificar nombres personalizados para los nodos utilizando add_node:
builder.add_node("my_node", step_1)
Tenga en cuenta que:
add_edgetoma los nombres de los nodos, que para las funciones es por defecto node.__name__.
Debemos especificar el punto de entrada del grafo. Para ello, añadimos una arista con el nodo INICIO .
El gráfico se detiene cuando no hay más nodos que ejecutar.
A continuación, compilamos nuestro grafo. Esto proporciona algunas comprobaciones básicas sobre la estructura del grafo (por ejemplo, la identificación de nodos huérfanos). Si añadiéramos persistencia a nuestra aplicación mediante un puntero de control , también se pasaría aquí.
graph = builder.compile()
LangGraph proporciona utilidades integradas para visualizar su grafo. Analicemos nuestra secuencia. Consulte Visualice su grafo para obtener más detalles sobre la visualización.

```python
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
Gráfico de secuencia de pasos
Procedamos con una invocación simple:
graph.invoke({"value_1": "c"})
{'value_1': 'a b', 'value_2': 10}
```

Tenga en cuenta que:
Iniciamos la invocación proporcionando un valor para una única clave de estado. Siempre debemos proporcionar un valor para al menos una clave.
El valor que le pasamos fue sobrescrito por el primer nodo.
El segundo nodo actualizó el valor.
El tercer nodo arrojó un valor diferente.
La notación abreviada integrada langgraph>=0.2.46 incluye una notación abreviada integrada add_sequencepara agregar secuencias de nodos. Puede compilar el mismo gráfico de la siguiente manera:

```python
builder = StateGraph(State).add_sequence([step_1, step_2, step_3])
builder.add_edge(START, "step_1")

graph = builder.compile()

graph.invoke({"value_1": "c"})
​```

Crear ramas
La ejecución paralela de nodos es esencial para acelerar el funcionamiento general del grafo. LangGraph ofrece soporte nativo para la ejecución paralela de nodos, lo que puede mejorar significativamente el rendimiento de los flujos de trabajo basados ​​en grafos. Esta paralelización se logra mediante mecanismos de ramificación (fan-out y fan-in), utilizando tanto aristas estándar como aristas condicionales (conditional_edges) . A continuación, se muestran algunos ejemplos de cómo agregar y crear flujos de datos ramificados que se adapten a sus necesidades.
​
Ejecutar los nodos del gráfico en paralelo
En este ejemplo, expandimos desde Node Ahasta B and Cy luego expandimos hacia adentro hasta D. Con nuestro estado, especificamos la operación de adición del reductor . Esto combinará o acumulará valores para la clave específica en el estado, en lugar de simplemente sobrescribir el valor existente. Para listas, esto significa concatenar la nueva lista con la lista existente. Consulte la sección anterior sobre reductores de estado para obtener más detalles sobre cómo actualizar el estado con reductores.

```python
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_node(d)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
Grafo de ejecución paralela
Con el reductor, se puede observar que los valores añadidos en cada nodo se acumulan.
graph.invoke({"aggregate": []}, {"configurable": {"thread_id": "foo"}})
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "D" to ['A', 'B', 'C']
```

En el ejemplo anterior, los nodos "b"y "c"se ejecutan simultáneamente en el mismo superpaso . Debido a que están en el mismo paso, el nodo "d"se ejecuta después de que ambos "b"y "c"hayan terminado.
Es importante destacar que las actualizaciones de un superpaso paralelo pueden no estar ordenadas de forma consistente. Si necesita un orden predeterminado y consistente para las actualizaciones de un superpaso paralelo, debe escribir las salidas en un campo separado del estado junto con un valor que indique el orden.
¿Manejo de excepciones?

LangGraph ejecuta los nodos dentro de los superpasos , lo que significa que, si bien las ramas paralelas se ejecutan en paralelo, todo el superpaso es transaccional . Si alguna de estas ramas genera una excepción, ninguna de las actualizaciones se aplica al estado (todo el superpaso falla).
Es importante destacar que, al usar un punto de control , los resultados de los nodos que se ejecutan correctamente dentro de un superpaso se guardan y no se repiten al reanudarse la ejecución.
Si tiene problemas de errores (quizás quiera manejar llamadas a la API inestables), LangGraph ofrece dos maneras de abordar esto:
Puedes escribir código Python estándar dentro de tu nodo para capturar y manejar excepciones.
Puedes configurar una política de reintentos para que el gráfico intente ejecutar nuevamente los nodos que generen ciertos tipos de excepciones. Solo se reintentan las ramas que fallan, por lo que no tienes que preocuparte por realizar trabajo redundante.
En conjunto, estas características permiten realizar una ejecución en paralelo y controlar completamente el manejo de excepciones.
Establecer la concurrencia máxima Puede controlar el número máximo de tareas concurrentes configurándolo max_concurrencyen la configuración al invocar el gráfico.
graph.invoke({"value_1": "c"}, {"configurable": {"max_concurrency": 10}})
​
Aplazar la ejecución del nodo
Aplazar la ejecución de un nodo resulta útil cuando se desea retrasar su ejecución hasta que se completen todas las demás tareas pendientes. Esto es especialmente relevante cuando las ramas tienen longitudes diferentes, algo común en flujos de trabajo como los de map-reduce.
El ejemplo anterior mostró cómo ramificar y concentrar cuando cada ruta constaba de un solo paso. Pero, ¿qué sucede si una rama tiene más de un paso? Agreguemos un nodo "b_2"a la "b"rama:

```python
import operator
from typing import Annotated, Any
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def b_2(state: State):
    print(f'Adding "B_2" to {state["aggregate"]}')
    return {"aggregate": ["B_2"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

def d(state: State):
    print(f'Adding "D" to {state["aggregate"]}')
    return {"aggregate": ["D"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(b_2)
builder.add_node(c)
builder.add_node(d, defer=True)
builder.add_edge(START, "a")
builder.add_edge("a", "b")
builder.add_edge("a", "c")
builder.add_edge("b", "b_2")
builder.add_edge("b_2", "d")
builder.add_edge("c", "d")
builder.add_edge("d", END)
graph = builder.compile()
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
Grafo de ejecución diferida
graph.invoke({"aggregate": []})
Adding "A" to []
Adding "B" to ['A']
Adding "C" to ['A']
Adding "B_2" to ['A', 'B', 'C']
Adding "D" to ['A', 'B', 'C', 'B_2']
```

En el ejemplo anterior, los nodos "b"y "c"se ejecutan simultáneamente en el mismo superpaso. Configuramos defer=Trueen el nodo dpara que no se ejecute hasta que todas las tareas pendientes hayan finalizado. En este caso, esto significa que "d"espera a ejecutarse hasta que toda la "b"rama haya finalizado.
​
Ramificación condicional
Si la ramificación debe variar en tiempo de ejecución según el estado, puede utilizar add_conditional_edgesel estado del grafo para seleccionar una o más rutas. Vea el ejemplo a continuación, donde el nodo agenera una actualización de estado que determina el siguiente nodo.

```python
import operator
from typing import Annotated, Literal, Sequence
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    aggregate: Annotated[list, operator.add]
    # Add a key to the state. We will set this key to determine
    # how we branch.
    which: str

def a(state: State):
    print(f'Adding "A" to {state["aggregate"]}')
    return {"aggregate": ["A"], "which": "c"}

def b(state: State):
    print(f'Adding "B" to {state["aggregate"]}')
    return {"aggregate": ["B"]}

def c(state: State):
    print(f'Adding "C" to {state["aggregate"]}')
    return {"aggregate": ["C"]}

builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)
builder.add_node(c)
builder.add_edge(START, "a")
builder.add_edge("b", END)
builder.add_edge("c", END)

def conditional_edge(state: State) -> Literal["b", "c"]:
    # Fill in arbitrary logic here that uses the state
    # to determine the next node
    return state["which"]

builder.add_conditional_edges("a", conditional_edge)

graph = builder.compile()
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
Grafo de ramificación condicional
result = graph.invoke({"aggregate": []})
print(result)
Adding "A" to []
Adding "C" to ['A']
{'aggregate': ['A', 'C'], 'which': 'c'}
```

Sus aristas condicionales pueden enrutar a múltiples nodos de destino. Por ejemplo:

```python
def route_bc_or_cd(state: State) -> Sequence[str]:
    if state["which"] == "cd":
        return ["c", "d"]
    return ["b", "c"]
​```

Map-Reduce y la API de envío
LangGraph admite map-reduce y otros patrones de ramificación avanzados mediante la API Send. Aquí tienes un ejemplo de cómo usarlo:

```python
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from typing_extensions import TypedDict, Annotated
import operator

class OverallState(TypedDict):
    topic: str
    subjects: list[str]
    jokes: Annotated[list[str], operator.add]
    best_selected_joke: str

def generate_topics(state: OverallState):
    return {"subjects": ["lions", "elephants", "penguins"]}

def generate_joke(state: OverallState):
    joke_map = {
        "lions": "Why don't lions like fast food? Because they can't catch it!",
        "elephants": "Why don't elephants use computers? They're afraid of the mouse!",
        "penguins": "Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice."
    }
    return {"jokes": [joke_map[state["subject"]]]}

def continue_to_jokes(state: OverallState):
    return [Send("generate_joke", {"subject": s}) for s in state["subjects"]]

def best_joke(state: OverallState):
    return {"best_selected_joke": "penguins"}

builder = StateGraph(OverallState)
builder.add_node("generate_topics", generate_topics)
builder.add_node("generate_joke", generate_joke)
builder.add_node("best_joke", best_joke)
builder.add_edge(START, "generate_topics")
builder.add_conditional_edges("generate_topics", continue_to_jokes, ["generate_joke"])
builder.add_edge("generate_joke", "best_joke")
builder.add_edge("best_joke", END)
graph = builder.compile()
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
Grafo MapReduce con ramificación
# Call the graph: here we call it to generate a list of jokes
for step in graph.stream({"topic": "animals"}):
    print(step)
{'generate_topics': {'subjects': ['lions', 'elephants', 'penguins']}}
{'generate_joke': {'jokes': ["Why don't lions like fast food? Because they can't catch it!"]}}
{'generate_joke': {'jokes': ["Why don't elephants use computers? They're afraid of the mouse!"]}}
{'generate_joke': {'jokes': ['Why don't penguins like talking to strangers at parties? Because they find it hard to break the ice.']}}
{'best_joke': {'best_selected_joke': 'penguins'}}
​```

Crear y controlar bucles
Al crear un grafo con un bucle, necesitamos un mecanismo para finalizar la ejecución. Lo más común es añadir una arista condicional que dirija al nodo FIN una vez que se cumpla alguna condición de terminación.
También puedes establecer el límite de recursión del grafo al invocarlo o procesarlo. El límite de recursión determina la cantidad de superpasos que el grafo puede ejecutar antes de generar un error. Obtén más información sobre el concepto de límite de recursión .
Consideremos un gráfico simple con un bucle para comprender mejor cómo funcionan estos mecanismos.
Para devolver el último valor de su estado en lugar de recibir un error de límite de recursión, consulte la siguiente sección .
Al crear un bucle, puede incluir un borde condicional que especifique una condición de terminación:

```python
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

def route(state: State) -> Literal["b", END]:
    if termination_condition(state):
        return END
    else:
        return "b"

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
```

Para controlar el límite de recursión, especifíquelo "recursionLimit"en la configuración. Esto generará una excepción GraphRecursionErrorque podrá capturar y manejar:

```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke(inputs, {"recursion_limit": 3})
except GraphRecursionError:
    print("Recursion Error")
```

Definamos un grafo con un bucle simple. Nótese que utilizamos una arista condicional para implementar una condición de terminación.

```python
import operator
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    # The operator.add reducer fn makes this append-only
    aggregate: Annotated[list, operator.add]

def a(state: State):
    print(f'Node A sees {state["aggregate"]}')
    return {"aggregate": ["A"]}

def b(state: State):
    print(f'Node B sees {state["aggregate"]}')
    return {"aggregate": ["B"]}

# Define nodes
builder = StateGraph(State)
builder.add_node(a)
builder.add_node(b)

# Define edges
def route(state: State) -> Literal["b", END]:
    if len(state["aggregate"]) < 7:
        return "b"
    else:
        return END

builder.add_edge(START, "a")
builder.add_conditional_edges("a", route)
builder.add_edge("b", "a")
graph = builder.compile()
from IPython.display import Image, display

display(Image(graph.get_graph().draw_mermaid_png()))
```

Grafo de bucle simple
Esta arquitectura es similar a la de un agente ReAct, donde node "a"es un modelo de llamada a herramientas y node "b"representa las herramientas.
En nuestra routearista condicional, especificamos que debemos terminar después de que la "aggregate"lista en el estado supere una longitud umbral.
Al invocar el gráfico, vemos que alternamos entre nodos "a"antes "b"de terminar una vez que alcanzamos la condición de terminación.

```python
graph.invoke({"aggregate": []})
Node A sees []
Node B sees ['A']
Node A sees ['A', 'B']
Node B sees ['A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B']
Node B sees ['A', 'B', 'A', 'B', 'A']
Node A sees ['A', 'B', 'A', 'B', 'A', 'B']
​```

Imponer un límite de recursión
En algunas aplicaciones, es posible que no tengamos la garantía de que alcanzaremos una condición de terminación determinada. En estos casos, podemos establecer el límite de recursión del grafo . Esto generará una excepción GraphRecursionErrordespués de un número determinado de superpasos . A continuación, podemos capturar y gestionar esta excepción:

```python
from langgraph.errors import GraphRecursionError

try:
    graph.invoke({"aggregate": []}, {"recursion_limit": 4})
except GraphRecursionError:
    print("Recursion Error")
Node A sees []
Node B sees ['A']
Node C sees ['A', 'B']
Node D sees ['A', 'B']
Node A sees ['A', 'B', 'C', 'D']
Recursion Error
Ejemplo extendido: devolver el estado al alcanzar el límite de recursión.

Ejemplo ampliado: bucles con ramificaciones
```
​
Asíncrono
El uso del paradigma de programación asíncrona puede generar mejoras significativas en el rendimiento al ejecutar código con restricciones de E/S de forma concurrente (por ejemplo, al realizar solicitudes API concurrentes a un proveedor de modelos de chat).
Para convertir una syncimplementación del gráfico en una asyncimplementación, necesitará:
Actualizar nodesel uso async defen lugar de def.
Actualiza el código interno para usarlo awaitcorrectamente.
Invoque el gráfico con .ainvokeo .astreamsegún se desee.
Debido a que muchos objetos LangChain implementan el Protocolo Runnable, que tiene asyncvariantes de todos los syncmétodos, por lo general es bastante rápido actualizar un syncgrafo a asyncotro.
Vea el ejemplo a continuación. Para demostrar las invocaciones asíncronas de los LLM subyacentes, incluiremos un modelo de chat: OpenAI, Antrópic, Azure, Google Gemini, AWS Bedrock

👉 Lea la documentación de integración del modelo de chat de OpenAI.
pip install -U "langchain[openai]"

modelo_de_chat_inicial

Clase modelo

```python
import os
from langchain.chat_models import init_chat_model

os.environ["OPENAI_API_KEY"] = "sk-..."

model = init_chat_model("gpt-5.2")
from langchain.chat_models import init_chat_model
from langgraph.graph import MessagesState, StateGraph

async def node(state: MessagesState):
    new_message = await llm.ainvoke(state["messages"])
    return {"messages": [new_message]}

builder = StateGraph(MessagesState).add_node(node).set_entry_point("node")
graph = builder.compile()

input_message = {"role": "user", "content": "Hello"}
result = await graph.ainvoke({"messages": [input_message]})
Transmisión asíncrona Consulte la guía de transmisión para ver ejemplos de transmisión asíncrona.
​```

Combine el flujo de control y las actualizaciones de estado conCommand
Puede resultar útil combinar el flujo de control (aristas) y las actualizaciones de estado (nodos). Por ejemplo, es posible que desee realizar actualizaciones de estado Y decidir a qué nodo ir a continuación en el MISMO nodo. LangGraph proporciona una forma de hacerlo devolviendo un objeto Command desde las funciones de los nodos:

```python
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        # state update
        update={"foo": "bar"},
        # control flow
        goto="my_other_node"
    )
```

A continuación, mostramos un ejemplo completo. Creemos un grafo simple con tres nodos: A, B y C. Primero, ejecutaremos el nodo A y, según el resultado del nodo A, decidiremos si pasar al nodo B o al nodo C.

```python
import random
from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START
from langgraph.types import Command

# Define graph state
class State(TypedDict):
    foo: str

# Define the nodes

def node_a(state: State) -> Command[Literal["node_b", "node_c"]]:
    print("Called A")
    value = random.choice(["b", "c"])
    # this is a replacement for a conditional edge function
    if value == "b":
        goto = "node_b"
    else:
        goto = "node_c"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        # this is the state update
        update={"foo": value},
        # this is a replacement for an edge
        goto=goto,
    )

def node_b(state: State):
    print("Called B")
    return {"foo": state["foo"] + "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": state["foo"] + "c"}
```

Ahora podemos crearlo StateGraphcon los nodos anteriores. ¡Observe que el grafo no tiene aristas condicionales para el enrutamiento! Esto se debe a que el flujo de control se define con Commanddentro de node_a.

```python
builder = StateGraph(State)
builder.add_edge(START, "node_a")
builder.add_node(node_a)
builder.add_node(node_b)
builder.add_node(node_c)
# NOTE: there are no edges between nodes A, B and C!

graph = builder.compile()
Es posible que hayas notado que usamos Commandcomo anotación de tipo de retorno, por ejemplo Command[Literal["node_b", "node_c"]]. Esto es necesario para la representación del gráfico y le indica a LangGraph que node_apuede navegar a node_by node_c.
from IPython.display import display, Image

display(Image(graph.get_graph().draw_mermaid_png()))
```

Navegación gráfica basada en comandos
Si ejecutamos el gráfico varias veces, veremos que toma diferentes caminos (A -> B o A -> C) dependiendo de la elección aleatoria en el nodo A.
graph.invoke({"foo": ""})
Called A
Called C
​
Navega a un nodo en un grafo padre.
Si está utilizando subgrafos , es posible que desee navegar desde un nodo dentro de un subgrafo a un subgrafo diferente (es decir, un nodo diferente en el grafo padre). Para ello, puede especificar graph=Command.PARENTen Command:
def my_node(state: State) -> Command[Literal["my_other_node"]]:
    return Command(
        update={"foo": "bar"},
        goto="other_subgraph",  # where `other_subgraph` is a node in the parent graph
        graph=Command.PARENT
    )
Vamos a demostrarlo usando el ejemplo anterior. Lo haremos transformando nodeAel ejemplo anterior en un grafo de un solo nodo que añadiremos como subgrafo a nuestro grafo padre.
Actualizaciones de estadoCommand.PARENT Cuando envías actualizaciones desde un nodo de subgrafo a un nodo de grafo padre para una clave que es compartida por los esquemas de estado del padre y del subgrafo , debes definir un reductor para la clave que estás actualizando en el estado del grafo padre. Consulta el ejemplo a continuación.

```python
import operator
from typing_extensions import Annotated

class State(TypedDict):
    # NOTE: we define a reducer here
    foo: Annotated[str, operator.add]

def node_a(state: State):
    print("Called A")
    value = random.choice(["a", "b"])
    # this is a replacement for a conditional edge function
    if value == "a":
        goto = "node_b"
    else:
        goto = "node_c"

    # note how Command allows you to BOTH update the graph state AND route to the next node
    return Command(
        update={"foo": value},
        goto=goto,
        # this tells LangGraph to navigate to node_b or node_c in the parent graph
        # NOTE: this will navigate to the closest parent graph relative to the subgraph
        graph=Command.PARENT,
    )

subgraph = StateGraph(State).add_node(node_a).add_edge(START, "node_a").compile()

def node_b(state: State):
    print("Called B")
    # NOTE: since we've defined a reducer, we don't need to manually append
    # new characters to existing 'foo' value. instead, reducer will append these
    # automatically (via operator.add)
    return {"foo": "b"}

def node_c(state: State):
    print("Called C")
    return {"foo": "c"}

builder = StateGraph(State)
builder.add_edge(START, "subgraph")
builder.add_node("subgraph", subgraph)
builder.add_node(node_b)
builder.add_node(node_c)

graph = builder.compile()
graph.invoke({"foo": ""})
Called A
Called C
​```

Utilice herramientas internas
Un caso de uso común es actualizar el estado del gráfico desde dentro de una herramienta. Por ejemplo, en una aplicación de atención al cliente, es posible que desee consultar la información del cliente en función de su número de cuenta o ID al inicio de la conversación. Para actualizar el estado del gráfico desde la herramienta, puede devolver Command(update={"my_custom_key": "foo", "messages": [...]})desde la herramienta:

```python
from langchain.tools import ToolRuntime

@tool
def lookup_user_info(runtime: ToolRuntime):
    """Use this to look up user information to better assist them with their questions."""
    user_info = get_user_info(runtime.server_info.user.identity)
    return Command(
        update={
            # update the state keys
            "user_info": user_info,
            # update the message history
            "messages": [ToolMessage("Successfully looked up user information", tool_call_id=runtime.tool_call_id)]
        }
    )
```

DEBE incluir messages(o cualquier clave de estado utilizada para el historial de mensajes) al Command.updateregresar Commandde una herramienta y la lista de mensajes en messagesDEBE contener un ToolMessage. Esto es necesario para que el historial de mensajes resultante sea válido (los proveedores de LLM requieren que los mensajes de IA con llamadas a herramientas vayan seguidos de los mensajes de resultado de la herramienta).
Si utiliza herramientas que actualizan el estado mediante Command, le recomendamos usar la versión predefinida ToolNode, que gestiona automáticamente los objetos que devuelven las herramientas Commandy los propaga al estado del gráfico. Si está escribiendo un nodo personalizado que llama a las herramientas, deberá propagar manualmente Commandlos objetos devueltos por las herramientas como actualización desde el nodo.
​
Visualiza tu gráfico
Aquí te mostramos cómo visualizar los gráficos que crees.
Puedes visualizar cualquier grafo arbitrario , incluido StateGraph .
Vamos a divertirnos dibujando fractales :).

```python
import random
from typing import Annotated, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

class State(TypedDict):
    messages: Annotated[list, add_messages]

class MyNode:
    def __init__(self, name: str):
        self.name = name
    def __call__(self, state: State):
        return {"messages": [("assistant", f"Called node {self.name}")]}

def route(state) -> Literal["entry_node", END]:
    if len(state["messages"]) > 10:
        return END
    return "entry_node"

def add_fractal_nodes(builder, current_node, level, max_level):
    if level > max_level:
        return
    # Number of nodes to create at this level
    num_nodes = random.randint(1, 3)  # Adjust randomness as needed
    for i in range(num_nodes):
        nm = ["A", "B", "C"][i]
        node_name = f"node_{current_node}_{nm}"
        builder.add_node(node_name, MyNode(node_name))
        builder.add_edge(current_node, node_name)
        # Recursively add more nodes
        r = random.random()
        if r > 0.2 and level + 1 < max_level:
            add_fractal_nodes(builder, node_name, level + 1, max_level)
        elif r > 0.05:
            builder.add_conditional_edges(node_name, route, node_name)
        else:
            # End
            builder.add_edge(node_name, END)

def build_fractal_graph(max_level: int):
    builder = StateGraph(State)
    entry_point = "entry_node"
    builder.add_node(entry_point, MyNode(entry_point))
    builder.add_edge(START, entry_point)
    add_fractal_nodes(builder, entry_point, 1, max_level)
    # Optional: set a finish point if required
    builder.add_edge(entry_point, END)  # or any specific node
    return builder.compile()

app = build_fractal_graph(3)
​
Sirena
También podemos convertir una clase de gráfico a la sintaxis de Mermaid.
print(app.get_graph().draw_mermaid())
%%{init: {'flowchart': {'curve': 'linear'}}}%%
graph TD;
    tart__([<p>__start__</p>]):::first
    ry_node(entry_node)
    e_entry_node_A(node_entry_node_A)
    e_entry_node_B(node_entry_node_B)
    e_node_entry_node_B_A(node_node_entry_node_B_A)
    e_node_entry_node_B_B(node_node_entry_node_B_B)
    e_node_entry_node_B_C(node_node_entry_node_B_C)
    nd__([<p>__end__</p>]):::last
    tart__ --> entry_node;
    ry_node --> __end__;
    ry_node --> node_entry_node_A;
    ry_node --> node_entry_node_B;
    e_entry_node_B --> node_node_entry_node_B_A;
    e_entry_node_B --> node_node_entry_node_B_B;
    e_entry_node_B --> node_node_entry_node_B_C;
    e_entry_node_A -.-> entry_node;
    e_entry_node_A -.-> __end__;
    e_node_entry_node_B_A -.-> entry_node;
    e_node_entry_node_B_A -.-> __end__;
    e_node_entry_node_B_B -.-> entry_node;
    e_node_entry_node_B_B -.-> __end__;
    e_node_entry_node_B_C -.-> entry_node;
    e_node_entry_node_B_C -.-> __end__;
    ssDef default fill:#f2f0ff,line-height:1.2
    ssDef first fill-opacity:0
    ssDef last fill:#bfb6fc
​```
Si se prefiere, podríamos convertir el gráfico en un .png. Aquí podríamos usar tres opciones:
Utilizando la API de Mermaid.ink (no requiere paquetes adicionales).
Usando Mermaid + Pyppeteer (requiere pip install pyppeteer)
Usando graphviz (que requiere pip install graphviz)
Usando Mermaid.Ink
Por defecto, draw_mermaid_png()utiliza la API de Mermaid.Ink para generar el diagrama.

```python
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

display(Image(app.get_graph().draw_mermaid_png()))
Visualización de gráficos fractales
Usando Mermaid + Pyppeteer
import nest_asyncio

nest_asyncio.apply()  # Required for Jupyter Notebook to run async functions

display(
    Image(
        app.get_graph().draw_mermaid_png(
            curve_style=CurveStyle.LINEAR,
            node_colors=NodeStyles(first="#ffdfba", last="#baffc9", default="#fad7de"),
            wrap_label_n_words=9,
            output_file_path=None,
            draw_method=MermaidDrawMethod.PYPPETEER,
            background_color="white",
            padding=10,
        )
    )
)
Usando Graphviz
try:
    display(Image(app.get_graph().draw_png()))
except ImportError:
    print(
        "You likely need to install dependencies for pygraphviz, see more here https://github.com/pygraphviz/pygraphviz/blob/main/INSTALL.txt"
    )
```