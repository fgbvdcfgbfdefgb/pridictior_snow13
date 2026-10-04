# Security

- No secret is required for Binance archive downloads or Bybit public market-data streams.
- Never commit tokens, exchange credentials, `.env`, private keys or Snowflake credentials.
- Supply any future private credential through the host's secret manager/environment, with least privilege and withdrawal disabled.
- Run the predictor separately from any order executor. The code in this repository has no order-placement implementation.
- Rotate credentials exposed in chat, terminal history or logs.
- Validate LFS artifact hashes before loading checkpoints; `torch.load` should only open trusted files.
