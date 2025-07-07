import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os

class PortfolioTracker:
    def __init__(self, portfolio_file='portfolio.json'):
        self.portfolio_file = portfolio_file
        self.portfolio = self.load_portfolio()
    
    def load_portfolio(self):
        """Load portfolio from file"""
        if os.path.exists(self.portfolio_file):
            try:
                with open(self.portfolio_file, 'r') as f:
                    return json.load(f)
            except:
                return self.get_default_portfolio()
        return self.get_default_portfolio()
    
    def save_portfolio(self):
        """Save portfolio to file"""
        with open(self.portfolio_file, 'w') as f:
            json.dump(self.portfolio, f, indent=2)
    
    def get_default_portfolio(self):
        """Get default portfolio structure"""
        return {
            'holdings': {},
            'cash': 10000.0,
            'transactions': [],
            'created_date': datetime.now().isoformat()
        }
    
    def add_stock(self, ticker, shares, price_per_share, date=None):
        """Add stock to portfolio"""
        if date is None:
            date = datetime.now().isoformat()
        
        ticker = ticker.upper()
        total_cost = shares * price_per_share
        
        if total_cost > self.portfolio['cash']:
            return False, "Insufficient cash"
        
        # Add to holdings
        if ticker in self.portfolio['holdings']:
            # Update existing holding
            current_shares = self.portfolio['holdings'][ticker]['shares']
            current_cost = self.portfolio['holdings'][ticker]['total_cost']
            
            new_shares = current_shares + shares
            new_cost = current_cost + total_cost
            avg_price = new_cost / new_shares
            
            self.portfolio['holdings'][ticker] = {
                'shares': new_shares,
                'avg_price': avg_price,
                'total_cost': new_cost
            }
        else:
            # New holding
            self.portfolio['holdings'][ticker] = {
                'shares': shares,
                'avg_price': price_per_share,
                'total_cost': total_cost
            }
        
        # Update cash
        self.portfolio['cash'] -= total_cost
        
        # Add transaction
        self.portfolio['transactions'].append({
            'date': date,
            'type': 'BUY',
            'ticker': ticker,
            'shares': shares,
            'price': price_per_share,
            'total': total_cost
        })
        
        self.save_portfolio()
        return True, "Stock added successfully"
    
    def sell_stock(self, ticker, shares, price_per_share, date=None):
        """Sell stock from portfolio"""
        if date is None:
            date = datetime.now().isoformat()
        
        ticker = ticker.upper()
        
        if ticker not in self.portfolio['holdings']:
            return False, "Stock not in portfolio"
        
        current_shares = self.portfolio['holdings'][ticker]['shares']
        
        if shares > current_shares:
            return False, "Insufficient shares"
        
        total_proceeds = shares * price_per_share
        
        # Update holdings
        if shares == current_shares:
            # Sell all shares
            del self.portfolio['holdings'][ticker]
        else:
            # Sell partial shares
            self.portfolio['holdings'][ticker]['shares'] -= shares
            # Note: We keep the same average price for simplicity
        
        # Update cash
        self.portfolio['cash'] += total_proceeds
        
        # Add transaction
        self.portfolio['transactions'].append({
            'date': date,
            'type': 'SELL',
            'ticker': ticker,
            'shares': shares,
            'price': price_per_share,
            'total': total_proceeds
        })
        
        self.save_portfolio()
        return True, "Stock sold successfully"
    
    def get_portfolio_summary(self, current_prices):
        """Get portfolio summary with current prices"""
        total_value = self.portfolio['cash']
        total_cost = 0
        holdings_summary = []
        
        for ticker, holding in self.portfolio['holdings'].items():
            current_price = current_prices.get(ticker, holding['avg_price'])
            current_value = holding['shares'] * current_price
            total_value += current_value
            total_cost += holding['total_cost']
            
            gain_loss = current_value - holding['total_cost']
            gain_loss_pct = (gain_loss / holding['total_cost']) * 100 if holding['total_cost'] > 0 else 0
            
            holdings_summary.append({
                'ticker': ticker,
                'shares': holding['shares'],
                'avg_price': holding['avg_price'],
                'current_price': current_price,
                'current_value': current_value,
                'total_cost': holding['total_cost'],
                'gain_loss': gain_loss,
                'gain_loss_pct': gain_loss_pct
            })
        
        portfolio_gain_loss = total_value - total_cost
        portfolio_gain_loss_pct = (portfolio_gain_loss / total_cost) * 100 if total_cost > 0 else 0
        
        return {
            'total_value': total_value,
            'total_cost': total_cost,
            'cash': self.portfolio['cash'],
            'gain_loss': portfolio_gain_loss,
            'gain_loss_pct': portfolio_gain_loss_pct,
            'holdings': holdings_summary
        }
    
    def get_transaction_history(self):
        """Get transaction history"""
        return sorted(self.portfolio['transactions'], key=lambda x: x['date'], reverse=True)
    
    def get_portfolio_allocation(self, current_prices):
        """Get portfolio allocation by stock"""
        summary = self.get_portfolio_summary(current_prices)
        allocation = []
        
        for holding in summary['holdings']:
            allocation.append({
                'ticker': holding['ticker'],
                'value': holding['current_value'],
                'percentage': (holding['current_value'] / summary['total_value']) * 100
            })
        
        # Add cash allocation
        if summary['cash'] > 0:
            allocation.append({
                'ticker': 'CASH',
                'value': summary['cash'],
                'percentage': (summary['cash'] / summary['total_value']) * 100
            })
        
        return sorted(allocation, key=lambda x: x['value'], reverse=True)
    
    def reset_portfolio(self):
        """Reset portfolio to default"""
        self.portfolio = self.get_default_portfolio()
        self.save_portfolio()
        return True, "Portfolio reset successfully" 