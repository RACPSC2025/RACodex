# 📦 SKILL.md Template

> Plantilla para crear nuevos packs de skills en RACodex.
> Copia este archivo, renómbralo y adapta según el dominio de tu pack.

---

## ¿Qué es este pack?

<!-- Describe en 2-3 líneas qué conocimiento aporta este pack -->

**Nombre del perfil:** `nombre-del-pack`  
**Autor:** Tu nombre o GitHub handle  
**Versión:** 0.1.0  
**Tags:** tag1, tag2, tag3

---

## ¿Cuándo se activa?

<!-- En qué situaciones el agente debería usar este pack -->

Este skill se activa cuando el usuario:
- Pide ayuda sobre [tema]
- Necesita [caso de uso]
- Trabaja con [tecnología/dominio]

---

## Contenido del Pack

<!-- Lista los archivos .md del pack con una línea de descripción cada uno -->

| Archivo | Descripción |
|---------|-------------|
| `fundamentals.md` | Conceptos base que todo developer necesita |
| `patterns.md` | Patrones de diseño aplicados a [dominio] |
| `best-practices.md` | Buenas prácticas y anti-patterns comunes |
| `reference.md` | Referencia rápida de comandos, APIs, configuraciones |

---

## Reglas de Comportamiento

<!-- Instrucciones de cómo el agente debe comportarse al usar este pack -->

1. **Siempre** haz X antes de Y
2. **Nunca** hagas Z sin confirmar
3. **Prioriza** A sobre B cuando haya conflicto
4. **Cita** fuentes oficiales cuando sea posible

---

## Ejemplos de Uso

<!-- 2-3 ejemplos concretos de consultas y cómo el pack guía la respuesta -->

### Ejemplo 1
**Usuario:** "¿Cómo estructuro un proyecto de [dominio]?"  
**Agente:** (Consulta `fundamentals.md`, aplica la arquitectura recomendada)  
**Resultado:** Estructura de carpetas con justificación técnica.

### Ejemplo 2
**Usuario:** "Mi [cosa] falla con [error]"  
**Agente:** (Consulta `best-practices.md`, aplica troubleshooting steps)  
**Resultado:** Diagnóstico paso a paso con referencias a la sección correcta.

---

## Dependencias

<!-- Otros packs o skills que este pack necesita -->

- Requiere: `general-dev/` (fundamentos compartidos)
- Opcional: `backend-python/` (si el proyecto usa Python)

---

## Changelog

| Versión | Fecha | Cambio |
|---------|-------|--------|
| 0.1.0 | 2026-04-08 | Versión inicial |

---

## Licencia

<!-- Mismo license que el proyecto RACodex (MIT) -->

MIT — Libre para usar, modificar y distribuir.
