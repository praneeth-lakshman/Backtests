import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Let's start with the absolute basics and build up

print("=== OPTIONS PRICING TUTORIAL ===")
print("Building intuition step by step\n")

# Step 1: What does an option payoff look like?
def option_payoff(stock_price, strike_price, option_type='call'):
    """
    Calculate option payoff at expiration
    This is the VALUE of the option when it expires
    """
    if option_type == 'call':
        # Call: right to BUY at strike price
        # Only valuable if stock > strike
        return max(stock_price - strike_price, 0)
    else:
        # Put: right to SELL at strike price  
        # Only valuable if stock < strike
        return max(strike_price - stock_price, 0)

# Example: Let's see call option payoffs
print("STEP 1: Understanding Option Payoffs")
print("-" * 40)

strike = 100
stock_prices = [80, 90, 95, 100, 105, 110, 120]

print(f"Call option with strike price ${strike}")
print("Stock Price | Option Value at Expiration")
for S in stock_prices:
    payoff = option_payoff(S, strike, 'call')
    print(f"   ${S:3d}     |        ${payoff:.0f}")

print(f"\nKey insight: Call option only has value when stock > ${strike}")
print("But what should we PAY for this option TODAY?\n")

# Step 2: Why is pricing hard?
print("STEP 2: Why Pricing is Challenging")
print("-" * 40)
print("The stock price is RANDOM. It could go to:")
print("- $120 (option worth $20)")  
print("- $80  (option worth $0)")
print("- Anything in between...")
print("So what's the 'fair' price today?\n")

# Step 3: Simple case - coin flip world
print("STEP 3: Simple Two-Outcome World")
print("-" * 40)
print("Let's say stock is $100 today, and in 1 year:")
print("- 50% chance it goes to $120")
print("- 50% chance it goes to $80")
print("Call option (strike $100) would be worth:")
print("- $20 if stock hits $120")
print("- $0 if stock hits $80")
print("Average payoff = 0.5 × $20 + 0.5 × $0 = $10")
print("But should we pay $10 today? NO!")
print("Money in the future is worth less (time value of money)")

risk_free_rate = 0.05  # 5% per year
discount_factor = np.exp(-risk_free_rate * 1)  # 1 year
fair_value = 10 * discount_factor
print(f"Fair value today = $10 × e^(-5% × 1 year) = ${fair_value:.2f}")
print("This is the BLACK-SCHOLES principle!\n")

# Step 4: The Black-Scholes formula (don't panic!)
print("STEP 4: The Black-Scholes Formula")
print("-" * 40)
print("Instead of just 2 outcomes, stocks can move continuously")
print("Black-Scholes assumes stock follows 'geometric Brownian motion'")
print("Fancy term for: random walk with drift and volatility")
print()

def black_scholes_call(S, K, T, r, sigma):
    """
    Black-Scholes call option price
    
    S = current stock price
    K = strike price  
    T = time to expiration (years)
    r = risk-free rate
    sigma = volatility (standard deviation of stock returns)
    """
    # These are the famous d1 and d2 from the formula
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # N(d) is cumulative normal distribution
    # This represents probabilities in the risk-neutral world
    call_price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    
    return call_price, d1, d2

# Example calculation
S = 100    # Stock price today
K = 100    # Strike price (at-the-money)
T = 0.25   # 3 months = 0.25 years
r = 0.05   # 5% risk-free rate
sigma = 0.2  # 20% annual volatility

price, d1, d2 = black_scholes_call(S, K, T, r, sigma)

print(f"Example: Stock=${S}, Strike=${K}, Time=3months, Vol=20%")
print(f"Black-Scholes call price: ${price:.2f}")
print(f"d1 = {d1:.3f}, d2 = {d2:.3f}")
print("d1 and d2 are like 'adjusted probabilities' in the formula\n")

# Step 5: What affects option prices?
print("STEP 5: What Makes Options More/Less Expensive?")
print("-" * 40)

base_params = [S, K, T, r, sigma]

print("Factor Analysis:")
print("1. VOLATILITY (sigma) - How much the stock jumps around")
vols = [0.1, 0.2, 0.3, 0.4]
print("Volatility | Option Price")
for vol in vols:
    params = base_params.copy()
    params[4] = vol
    price, _, _ = black_scholes_call(*params)
    print(f"   {vol:3.0%}     |   ${price:.2f}")
print("Higher volatility = Higher option price (more chance of big moves)\n")

print("2. TIME TO EXPIRATION (T) - How long until option expires")
times = [1/12, 3/12, 6/12, 12/12]  # 1, 3, 6, 12 months
print("Time (months) | Option Price")
for time in times:
    params = base_params.copy()
    params[2] = time
    price, _, _ = black_scholes_call(*params)
    print(f"     {time*12:.0f}       |   ${price:.2f}")
print("More time = Higher option price (more opportunities)\n")

print("3. DISTANCE FROM STRIKE (Moneyness)")
strikes = [90, 95, 100, 105, 110]
print("Strike | Option Price | In/Out of Money")
for strike in strikes:
    params = base_params.copy()
    params[1] = strike
    price, _, _ = black_scholes_call(*params)
    if strike < S:
        status = "In-the-money"
    elif strike == S:
        status = "At-the-money"
    else:
        status = "Out-of-the-money"
    print(f"  ${strike:3d}  |    ${price:.2f}     | {status}")
print()

# Step 6: Visualizing option behavior
print("STEP 6: Visualizing How Options Behave")
print("-" * 40)

# Create plots to show option behavior
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Payoff diagram
stock_range = np.linspace(80, 120, 100)
call_payoffs = [option_payoff(s, 100, 'call') for s in stock_range]
put_payoffs = [option_payoff(s, 100, 'put') for s in stock_range]

ax1.plot(stock_range, call_payoffs, 'b-', label='Call Payoff', linewidth=2)
ax1.plot(stock_range, put_payoffs, 'r-', label='Put Payoff', linewidth=2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=100, color='k', linestyle='--', alpha=0.3, label='Strike')
ax1.set_xlabel('Stock Price at Expiration')
ax1.set_ylabel('Option Payoff')
ax1.set_title('Option Payoffs at Expiration')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Effect of volatility
vols = np.linspace(0.1, 0.5, 50)
vol_prices = []
for vol in vols:
    price, _, _ = black_scholes_call(100, 100, 0.25, 0.05, vol)
    vol_prices.append(price)

ax2.plot(vols * 100, vol_prices, 'g-', linewidth=2)
ax2.set_xlabel('Volatility (%)')
ax2.set_ylabel('Call Option Price')
ax2.set_title('Option Price vs Volatility')
ax2.grid(True, alpha=0.3)

# Plot 3: Time decay
times = np.linspace(0.01, 1, 50)
time_prices = []
for t in times:
    price, _, _ = black_scholes_call(100, 100, t, 0.05, 0.2)
    time_prices.append(price)

ax3.plot(times * 12, time_prices, 'orange', linewidth=2)
ax3.set_xlabel('Months to Expiration')
ax3.set_ylabel('Call Option Price')
ax3.set_title('Time Decay (Theta)')
ax3.grid(True, alpha=0.3)

# Plot 4: Price vs stock price
stock_prices = np.linspace(80, 120, 100)
current_prices = []
for s in stock_prices:
    price, _, _ = black_scholes_call(s, 100, 0.25, 0.05, 0.2)
    current_prices.append(price)

ax4.plot(stock_prices, current_prices, 'purple', linewidth=2, label='Current Value')
ax4.plot(stock_range, call_payoffs, 'b--', label='Payoff at Expiration', alpha=0.7)
ax4.set_xlabel('Stock Price')
ax4.set_ylabel('Call Option Value')
ax4.set_title('Current Value vs Payoff at Expiration')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Key insights from the plots:")
print("1. Options have curved value (not linear like stocks)")
print("2. Higher volatility = higher option prices")
print("3. Options lose value over time (time decay)")
print("4. Current option value > payoff today (time value)\n")

# Step 7: Greeks - Risk sensitivities
print("STEP 7: The Greeks - Managing Risk")
print("-" * 40)
print("'Greeks' measure how option prices change with different factors")
print()

def calculate_delta(S, K, T, r, sigma):
    """Delta: How much option price changes per $1 stock move"""
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def calculate_gamma(S, K, T, r, sigma):
    """Gamma: How much delta changes per $1 stock move"""
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

def calculate_theta(S, K, T, r, sigma):
    """Theta: How much option loses per day"""
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) - r * K * np.exp(-r*T) * norm.cdf(d2)
    return theta / 365  # Convert to per day

# Example Greeks calculation
delta = calculate_delta(100, 100, 0.25, 0.05, 0.2)
gamma = calculate_gamma(100, 100, 0.25, 0.05, 0.2)
theta = calculate_theta(100, 100, 0.25, 0.05, 0.2)

print(f"For our example option (S=$100, K=$100, 3 months, 20% vol):")
print(f"Delta = {delta:.3f} (if stock goes up $1, option goes up $0.{delta*100:.0f})")
print(f"Gamma = {gamma:.4f} (delta changes by this amount per $1 stock move)")
print(f"Theta = ${theta:.3f} (option loses this much value per day)")
print()

print("=== SUMMARY ===")
print("You now understand:")
print("1. Options give you rights, not obligations")
print("2. Payoffs depend on where stock ends up")
print("3. Current prices account for ALL possible outcomes") 
print("4. Black-Scholes gives us the 'fair' price")
print("5. Volatility and time increase option value")
print("6. Greeks help us manage risk")
print()
print("Next: Let's code a simple version together!")