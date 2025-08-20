import { render, screen, waitFor } from '@testing-library/react'
import Dashboard from '../pages/dashboard'

// Mock fetch
global.fetch = jest.fn()

describe('Dashboard', () => {
  beforeEach(() => {
    fetch.mockClear()
  })

  it('renders dashboard components', async () => {
    // Mock API responses
    fetch
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { id: 1, name: 'Apple Inc.', ticker: 'AAPL' },
          { id: 2, name: 'Microsoft Corporation', ticker: 'MSFT' }
        ]
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { id: 1, credit_score: 750, risk_category: 'A', score_date: '2024-01-01', confidence: 0.85 }
        ]
      })
      .mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          feature_contributions: [
            { feature: 'revenue_growth', contribution: 0.15, description: 'Revenue Growth Rate' },
            { feature: 'debt_ratio', contribution: -0.08, description: 'Debt to Equity Ratio' }
          ]
        })
      })

    render(<Dashboard />)

    // Check if main components are rendered
    expect(screen.getByText('Credit Intelligence Dashboard')).toBeInTheDocument()
    
    // Wait for data to load
    await waitFor(() => {
      expect(screen.getByText('Score Overview')).toBeInTheDocument()
    })
  })

  it('handles API errors gracefully', async () => {
    fetch.mockRejectedValueOnce(new Error('API Error'))

    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByText('Credit Intelligence Dashboard')).toBeInTheDocument()
    })
  })
})
