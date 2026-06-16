"""SQLite logging of everything: agents, skills, rewards, generations, experiments,
and skill-propagation events. One database per run under results/."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT, config TEXT
);
CREATE TABLE IF NOT EXISTS generations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT, generation INTEGER,
    avg_fitness REAL, best_fitness REAL, avg_solved REAL,
    culture_size INTEGER, avg_skills REAL, avg_difficulty_solved REAL,
    metrics TEXT
);
CREATE TABLE IF NOT EXISTS agents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT, agent_id TEXT, generation INTEGER, parents TEXT,
    specialization TEXT, lifetime_reward REAL, tasks_attempted INTEGER,
    tasks_solved INTEGER, reputation REAL, n_known_skills INTEGER,
    n_contributions INTEGER, n_taught INTEGER
);
CREATE TABLE IF NOT EXISTS skills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT, generation INTEGER, name TEXT, program TEXT,
    creator TEXT, preconditions TEXT, success_rate REAL, usage_count INTEGER,
    reputation REAL, adoption INTEGER, complexity INTEGER
);
CREATE TABLE IF NOT EXISTS propagation (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT, generation INTEGER, program TEXT,
    from_agent TEXT, to_agent TEXT
);
CREATE TABLE IF NOT EXISTS rewards (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT, generation INTEGER, agent_id TEXT,
    task TEXT, difficulty INTEGER, reward REAL, solved INTEGER
);
"""


class Database:
    def __init__(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def log_experiment(self, name, config: dict):
        self.conn.execute("INSERT INTO experiments(name, config) VALUES (?,?)",
                          (name, json.dumps(config)))
        self.conn.commit()

    def log_generation(self, experiment, generation, stats: dict):
        self.conn.execute(
            """INSERT INTO generations(experiment, generation, avg_fitness,
               best_fitness, avg_solved, culture_size, avg_skills,
               avg_difficulty_solved, metrics) VALUES (?,?,?,?,?,?,?,?,?)""",
            (experiment, generation, stats.get("avg_fitness"),
             stats.get("best_fitness"), stats.get("avg_solved"),
             stats.get("culture_size"), stats.get("avg_skills"),
             stats.get("avg_difficulty_solved"), json.dumps(stats)))
        self.conn.commit()

    def log_agent(self, experiment, row: dict):
        self.conn.execute(
            """INSERT INTO agents(experiment, agent_id, generation, parents,
               specialization, lifetime_reward, tasks_attempted, tasks_solved,
               reputation, n_known_skills, n_contributions, n_taught)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment, row["id"], row["generation"], row["parents"],
             row["specialization"], row["lifetime_reward"],
             row["tasks_attempted"], row["tasks_solved"], row["reputation"],
             row["n_known_skills"], row["n_contributions"], row["n_taught"]))

    def log_skill(self, experiment, generation, row: dict):
        self.conn.execute(
            """INSERT INTO skills(experiment, generation, name, program, creator,
               preconditions, success_rate, usage_count, reputation, adoption,
               complexity) VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (experiment, generation, row["name"], row["program"], row["creator"],
             row["preconditions"], row["success_rate"], row["usage_count"],
             row["reputation"], row["adoption"], row["complexity"]))

    def log_propagation(self, experiment, generation, program, from_agent, to_agent):
        self.conn.execute(
            """INSERT INTO propagation(experiment, generation, program, from_agent,
               to_agent) VALUES (?,?,?,?,?)""",
            (experiment, generation, program, from_agent, to_agent))

    def log_reward(self, experiment, generation, agent_id, task, difficulty,
                   reward, solved):
        self.conn.execute(
            """INSERT INTO rewards(experiment, generation, agent_id, task,
               difficulty, reward, solved) VALUES (?,?,?,?,?,?,?)""",
            (experiment, generation, agent_id, task, difficulty, reward,
             int(solved)))

    def commit(self):
        self.conn.commit()

    def close(self):
        self.conn.commit()
        self.conn.close()
