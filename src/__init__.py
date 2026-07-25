"""LIGERBOT — event-driven algorithmic trading bot.

Five decoupled microservice modules communicating over a Redis Streams event bus:

  1. data_ingestion   — Kotak Neo WebSocket  -> market_ticks
  3. strategy_engine  — market_ticks         -> trade_signals
  4. risk_manager     — trade_signals        -> approved_orders
  5. execution_engine — approved_orders      -> filled_orders (broker)
  6. storage_logger   — all streams          -> InfluxDB
"""

__version__ = "0.1.0"
