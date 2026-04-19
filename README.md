#  Lane-Aware Multi-Robot Traffic Control System

A simulation-based system for coordinating multiple autonomous robots in a warehouse-like grid while avoiding collisions, minimizing deadlocks, and maximizing throughput.

---

##  Overview

In modern warehouses, multiple robots share narrow pathways and intersections. Naive shortest-path routing leads to **deadlocks, congestion, and starvation**.

This project implements a **lane-aware, reservation-based traffic control system** that enables robots to:

* Navigate efficiently using A* path planning
* Avoid collisions through spatial reservation
* Detect and resolve deadlocks using graph theory
* Learn from past conflicts to prevent repeated failures

---

##  Key Objectives

*  Eliminate infinite deadlock loops
*  Improve throughput (tasks completed per tick)
*  Introduce memory-aware routing
*  Simulate realistic traffic behavior (like road systems)

---

##  Core Concepts

### 1. A* Path Planning

Each robot computes the shortest path to its destination using A*.

However, instead of blindly following shortest paths, the system modifies costs dynamically based on:

* congestion
* conflict history
* deadlock penalties

---

### 2. Reservation-Based Movement (Traffic Control)

A strict **3-stage locking pipeline** ensures safe movement:

1. **Node Lock (Origin)**
   Robot occupies its current node

2. **Edge + Target Lock (Before Moving)**
   Robot must reserve:

   * the edge it wants to enter
   * the destination node

   If unavailable → robot waits

3. **Release Mechanism**

   * Once movement starts → origin node is freed
   * On arrival → destination becomes occupied

👉 This prevents head-on collisions and intersection conflicts.

---

### 3. Deadlock Detection (Tarjan SCC)

We model robot dependencies as a **wait-for graph**:

* Robot A waiting on B → edge A → B

Using **Tarjan’s Strongly Connected Components algorithm**, we detect cycles:

* A → B → C → A ⇒ DEADLOCK

---

### 4. Deadlock Resolution + Memory

Instead of repeatedly breaking deadlocks randomly:

* We **penalize edges involved in cycles**
* Penalties accumulate over time
* A* avoids those edges in future paths

👉 This converts the system from **reactive → adaptive**

---

### 5. Starvation Prevention

If a robot waits too long:

* Its priority increases
* It eventually gets access to intersections

This ensures fairness across all robots.

---

##  Implementation Details

* Language: **Python**
* Algorithmic components:

  * A* Search
  * Tarjan SCC (cycle detection)
  * Heuristic cost adjustments
* Architecture:

  * `TrafficController` → manages reservations, deadlocks
  * `PathPlanner` → A* with dynamic penalties
  * `Robot` → state machine (MOVING, WAITING, DONE)

---

## 📊 Results

| Metric           | Results          |
| ---------------- | -------------------- |
| Completed Robots | **6 / 10**           |
| Deadlocks        |**20**               |
| Throughput       | **0.02 robots/tick** |
| Infinite Loops   | **Eliminated**     |

---

## How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/multi-robot-traffic-system.git
cd multi-robot-traffic-system
```

### 2. Run the simulation

```bash
python multi_robot_sim.py
```

(Optional headless mode)

```bash
python headless_sim.py
```

---

##  Output

The simulation provides:

*  Live grid visualization
*  Throughput graph
*  Robot wait times
*  Final summary including:

  * completed robots
  * deadlocks
  * emergency stops

---


##  Key Insight

> Pure shortest-path planning fails in multi-agent systems.
> **Coordination requires both space AND time awareness.**

---

##  Future Improvements

* Time-expanded planning (space-time A*)
* Multi-agent pathfinding (CBS / MAPF)
* Reinforcement learning for adaptive routing
* Real-world robot integration

---

##  Conclusion

This project demonstrates how combining:

* path planning
* graph theory
* reservation systems

can transform a chaotic multi-robot environment into a **stable, efficient traffic system**.

---

## Author

Developed as part of a hackathon project focused on intelligent multi-agent systems.

---
