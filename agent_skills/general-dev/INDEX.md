# 🔧 General Developer — Skill Pack

> Perfil base de desarrollo: fundamentos de ingeniería, patrones de diseño, system design,
> bases de datos, DevOps y cultura de ingeniería. Se activa cuando el usuario no selecciona
> un perfil específico.

---

## ¿Cuándo se activa este pack?

Este skill se activa cuando el usuario:
- No ha seleccionado un perfil específico de agente
- Necesita fundamentos de ingeniería agnósticos al lenguaje
- Busca patrones de diseño, SOLID, o arquitecturas backend
- Requiere guidance en system design o decisiones técnicas de alto nivel
- Trabaja con bases de datos, DevOps, o cultura de equipo de ingeniería

---

## Contenido del Pack

| Archivo | Descripción |
|---------|-------------|
| `engineering-fundamentals.md` | Referencia completa: DSA, patrones de diseño, arquitecturas backend, system design, bases de datos, DevOps |

### engineering-fundamentals.md incluye:
- **DSA** — Complejidad algorítmica, estructuras de datos, algoritmos esenciales
- **Patrones de Diseño** — SOLID, creacionales, estructurales, comportamiento, modernos
- **Arquitecturas Backend** — Hexagonal, microservicios, event-driven, CQRS, saga
- **System Design** — Framework de decisiones, CAP theorem, escalado, cache strategies
- **Bases de Datos** — SQL vs NoSQL, cuándo usar cada una, diseño de esquema
- **DevOps y Cultura** — DORA metrics, principios de ingeniería, automatización

---

## Reglas de Comportamiento

1. **Siempre** aplica SOLID antes de sugerir cualquier patrón de diseño
2. **Nunca** recomiendes microservicios sin que el usuario tenga una razón concreta medible
3. **Siempre** empieza con monolito modular bien estructurado — extrae servicios solo cuando sea necesario
4. **Prioriza** simplicidad sobre complejidad — la solución más simple que resuelva el problema
5. **Cita** la sección específica de `engineering-fundamentals.md` cuando apliques un concepto
6. **Pregunta** "¿Tenemos ese problema ahora?" antes de sugerir tecnologías complejas

---

## Ejemplos de Uso

### Ejemplo 1: Decidir arquitectura
**Usuario:** "¿Debería usar microservicios para mi startup?"  
**Agente:** (Consulta `engineering-fundamentals.md` → sección de Microservicios)  
**Resultado:** Checklist de extracción de microservicios + recomendación de empezar con monolito modular.

### Ejemplo 2: Elegir base de datos
**Usuario:** "¿Qué DB uso para mi proyecto?"  
**Agente:** (Consulta `engineering-fundamentals.md` → sección de Bases de Datos)  
**Resultado:** Tabla de comparación SQL vs NoSQL + recomendación basada en el caso de uso.

### Ejemplo 3: Mejorar código
**Usuario:** "Mi clase tiene 500 líneas y 20 métodos"  
**Agente:** (Consulta `engineering-fundamentals.md` → Principio de Responsabilidad Única)  
**Resultado:** Identifica God class + recomendación de dividir en servicios enfocados.

---

## Dependencias

- Ninguna — este es el pack base/fallback
- Otros packs pueden requerir este como dependencia base

---

## Changelog

| Versión | Fecha | Cambio |
|---------|-------|--------|
| 1.0.0 | 2026-04-08 | Versión inicial — fundamentos de ingeniería |

---

## Licencia

MIT — Libre para usar, modificar y distribuir.
