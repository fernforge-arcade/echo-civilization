"""Environment 4 — Social World (a Lewis signalling game).

Multiple agents communicate using meaningless symbols (0..M-1). No meanings are
given; agents must *agree on a protocol* through repeated interaction. A speaker
sees a hidden concept and emits a symbol; a listener sees the symbol and guesses
the concept. Correct guesses reward both (cooperation improves shared survival).

Each agent keeps reward-weighted association tables (Roth–Erev reinforcement) for
speaking (concept->symbol) and listening (symbol->concept). We measure whether a
stable communication protocol emerges and whether it raises population success.
"""

from __future__ import annotations

import numpy as np


class Signaller:
    """A simple emergent-communication learner attached to an agent."""

    def __init__(self, n_concepts: int, n_symbols: int, rng: np.random.Generator):
        self.nc = n_concepts
        self.ns = n_symbols
        self.rng = rng
        # association weights, start uniform (no built-in meaning)
        self.speak = np.ones((n_concepts, n_symbols))
        self.listen = np.ones((n_symbols, n_concepts))

    def _sample(self, weights):
        p = weights / weights.sum()
        return int(self.rng.choice(len(weights), p=p))

    def say(self, concept: int) -> int:
        return self._sample(self.speak[concept])

    def interpret(self, symbol: int) -> int:
        return self._sample(self.listen[symbol])

    def reinforce(self, concept, symbol, listener_guess, reward):
        if reward > 0:
            self.speak[concept, symbol] += reward
            self.listen[symbol, listener_guess] += reward

    def protocol(self):
        """Greedy concept->symbol map (the agent's current 'language')."""
        return tuple(int(np.argmax(self.speak[c])) for c in range(self.nc))


class SocialWorld:
    name = "social"

    def __init__(self, rng: np.random.Generator, n_concepts: int = 4,
                 n_symbols: int = 4):
        self.rng = rng
        self.n_concepts = n_concepts
        self.n_symbols = n_symbols

    def attach(self, agents):
        sig = {}
        for a in agents:
            sig[a.id] = Signaller(self.n_concepts, self.n_symbols, self.rng)
        return sig

    def play_round(self, agents, signallers) -> dict:
        """Random speaker/listener pairs play one referential game each."""
        successes = 0
        n = len(agents)
        order = list(range(n))
        self.rng.shuffle(order)
        pairs = [(order[i], order[(i + 1) % n]) for i in range(n)]
        for si, li in pairs:
            speaker, listener = agents[si], agents[li]
            concept = int(self.rng.integers(0, self.n_concepts))
            symbol = signallers[speaker.id].say(concept)
            guess = signallers[listener.id].interpret(symbol)
            reward = 1.0 if guess == concept else 0.0
            signallers[speaker.id].reinforce(concept, symbol, guess, reward)
            signallers[listener.id].reinforce(concept, symbol, guess, reward)
            if reward > 0:
                successes += 1
                speaker.befriend(listener.id, 0.05)
                listener.befriend(speaker.id, 0.05)
                speaker.lifetime_reward += 0.5
                listener.lifetime_reward += 0.5
        return {"accuracy": successes / len(pairs)}

    def protocol_consistency(self, agents, signallers) -> float:
        """Fraction of agents agreeing on the most common symbol per concept —
        a measure of how shared the emergent language is."""
        protos = np.array([signallers[a.id].protocol() for a in agents])
        agree = 0
        for c in range(self.n_concepts):
            col = protos[:, c]
            counts = np.bincount(col, minlength=self.n_symbols)
            agree += counts.max()
        return agree / (len(agents) * self.n_concepts)

    def run(self, agents, rounds: int = 60) -> dict:
        signallers = self.attach(agents)
        history = []
        consistency = []
        for r in range(rounds):
            res = self.play_round(agents, signallers)
            history.append(res["accuracy"])
            consistency.append(self.protocol_consistency(agents, signallers))
        return {
            "accuracy_curve": history,
            "consistency_curve": consistency,
            "final_accuracy": history[-1] if history else 0.0,
            "final_consistency": consistency[-1] if consistency else 0.0,
        }
