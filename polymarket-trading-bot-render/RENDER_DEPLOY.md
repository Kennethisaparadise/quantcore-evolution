# Master Trading System - Render Deployment

## Quick Deploy to Render (Free Tier)

### Step 1: Push to GitHub

```bash
cd polymarket-trading-bot-render
git add .
git commit -m "Ready for Render deployment"
git remote add origin https://github.com/YOUR_USERNAME/polymarket-trading-bot-render.git
git push -u origin main
```

### Step 2: Create Render Service

1. Go to [dashboard.render.com](https://dashboard.render.com)
2. Click **New +** → **Web Service**
3. Connect your GitHub repository
4. Configure:
   - **Name**: `polymarket-trading-bot`
   - **Branch**: `main`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python src/master_trading_system.py`
   - **Plan**: Free

### Step 3: Environment Variables

Add these in Render dashboard:

```env
# Trading System
INITIAL_CAPITAL=10000
SCAN_INTERVAL_SECONDS=300
CONFIDENCE_THRESHOLD=0.65
ALIGNMENT_THRESHOLD=0.70

# Telegram (optional)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
TELEGRAM_ENABLED=true

# Database (optional)
DATABASE_URL=sqlite:///trading.db
```

### Step 4: Health Check

Render needs a health check endpoint. The system sends periodic heartbeats.

---

## Render Free Tier Limitations

| Resource | Free Limit | Our Usage |
|----------|-----------|-----------|
| **Hours/month** | 750 | ~720 (always on) |
| **Memory** | 512 MB | ~200 MB |
| **CPU** | Shared | Sufficient |
| **Disk** | 1 GB | ~50 MB |
| **Sleep** | 15 min inactivity | We prevent this! |

### How We Stay Awake

Our system scans every 5 minutes, so it never sleeps on free tier:

```python
SCAN_INTERVAL_SECONDS = 300  # 5 minutes
```

Every scan keeps the service active!

---

## Scaling Options

### Free (Current)
- 750 hours/month
- Always on with 5-min scans
- Sufficient for testing

### Paid ($7/month)
- 100% always on
- More memory
- Better for live trading

---

## Files Needed for Render

```
polymarket-trading-bot-render/
├── requirements.txt
├── src/
│   ├── master_trading_system.py
│   ├── tri_rhythm_calculator.py
│   ├── multi_timeframe_scanner.py
│   ├── guide_rail_hft.py
│   ├── gann_calculator.py
│   ├── polymarket_connector.py
│   └── telegram_notifications.py
└── render.yaml (optional service configuration)
```

---

## Monitoring

### View Logs
```bash
# In Render dashboard
# OR
curl https://polymarket-trading-bot.onrender.com/logs
```

### Check Status
```bash
curl https://polymarket-trading-bot.onrender.com/health
```

---

## Telegram Alerts

1. Create bot: Message @BotFather on Telegram
2. Get chat ID: Message @userinfobot
3. Add credentials to Render environment variables

---

## Troubleshooting

### Service Won't Start
- Check logs in Render dashboard
- Verify `requirements.txt` is correct
- Ensure all imports work locally first

### Memory Issues
- Reduce `MAX_OPEN_POSITIONS`
- Increase scan interval
- Use smaller market list

### API Rate Limits
- Polymarket has rate limits
- System handles retries automatically
- Reduce `SCAN_INTERVAL_SECONDS` if needed

---

## Production Checklist

Before going live:

- [ ] Test on free tier for 7 days
- [ ] Verify Telegram notifications work
- [ ] Check daily P&L is positive
- [ ] Set proper stop losses
- [ ] Start with small capital ($1000-5000)
- [ ] Monitor for 2 weeks before scaling up
- [ ] Document any issues

---

## Important Disclaimers

⚠️ **Trading involves risk**

- Past performance ≠ future results
- Never trade more than you can afford to lose
- This is algorithmic trading - bugs can cause losses
- Always use proper risk management
- Consider paper trading first

⚠️ **Free tier limitations**

- No guaranteed uptime
- May restart occasionally
- Not suitable for high-frequency trading
- For education/demo purposes

---

## Support

- Issues: Open GitHub issue
- Docs: See README.md
- Telegram: Check your configured chat
