# Pokémon Showdown Strategy Bot

A Python project to create a machine learning bot that can play the competitive turn-based game pokemonshowdown.com, specifically the format "Gen 9 Random Battle."

The bot transforms raw battle replay logs into structured, turn-by-turn state-action data and trains predictive models through supervised learning to select moves.

This is a work in progress. Unfortunately, the official Showdown replay logs don't give full access to battle state (namely unrevealed pokemon, moves, and items), so I am planning on refining the model through reinforcement learning.

## Features

- Custom parser to download, inspect, and normalize replay logs
- Feature engineering pipeline to process state-action pairs
- Current model classifiers: HistGradientBoosting, LightGBM, Random Forest
- Hierarchical model architecture
- Hyperparameter tuning with cross-validation

