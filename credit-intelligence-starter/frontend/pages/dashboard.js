import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, 
  BarChart, Bar, PieChart, Pie, Cell, AreaChart, Area, ScatterPlot, Scatter,
  RadialBarChart, RadialBar, TreeMap, ComposedChart
} from 'recharts';
import styled from 'styled-components';

// Styled Components
const DashboardContainer = styled.div`
  display: flex;
  min-height: 100vh;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
`;

const Sidebar = styled.div`
  width: 280px;
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-right: 1px solid rgba(255, 255, 255, 0.2);
  padding: 20px;
  overflow-y: auto;
  box-shadow: 2px 0 10px rgba(0, 0, 0, 0.1);
`;

const MainContent = styled.div`
  flex: 1;
  padding: 20px;
  overflow-y: auto;
`;

const Header = styled.div`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
`;

const TabContainer = styled.div`
  display: flex;
  gap: 10px;
  margin-bottom: 20px;
`;

const Tab = styled.button`
  padding: 12px 24px;
  border: none;
  border-radius: 25px;
  background: ${props => props.active ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(255, 255, 255, 0.8)'};
  color: ${props => props.active ? 'white' : '#333'};
  cursor: pointer;
  font-weight: 600;
  transition: all 0.3s ease;
  
  &:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  }
`;

const Card = styled.div`
  background: rgba(255, 255, 255, 0.95);
  backdrop-filter: blur(10px);
  border-radius: 15px;
  padding: 20px;
  margin-bottom: 20px;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
  border: 1px solid rgba(255, 255, 255, 0.2);
`;

const GridContainer = styled.div`
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
  gap: 20px;
  margin-bottom: 20px;
`;

const MetricCard = styled(Card)`
  text-align: center;
  background: ${props => {
    if (props.risk === 'low') return 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)';
    if (props.risk === 'medium') return 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)';
    if (props.risk === 'high') return 'linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%)';
    return 'linear-gradient(135deg, #a8edea 0%, #fed6e3 100%)';
  }};
  color: white;
  min-height: 120px;
  display: flex;
  flex-direction: column;
  justify-content: center;
`;

const CompanyList = styled.div`
  max-height: 400px;
  overflow-y: auto;
`;

const CompanyItem = styled.div`
  padding: 12px;
  margin: 8px 0;
  border-radius: 10px;
  cursor: pointer;
  background: ${props => props.selected ? 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' : 'rgba(255, 255, 255, 0.5)'};
  color: ${props => props.selected ? 'white' : '#333'};
  transition: all 0.3s ease;
  border: 1px solid rgba(0, 0, 0, 0.1);
  
  &:hover {
    transform: translateX(5px);
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
  }
`;

const FilterContainer = styled.div`
  display: flex;
  gap: 15px;
  margin-bottom: 20px;
  flex-wrap: wrap;
`;

const FilterSelect = styled.select`
  padding: 10px 15px;
  border: 1px solid rgba(0, 0, 0, 0.2);
  border-radius: 10px;
  background: rgba(255, 255, 255, 0.9);
  font-size: 14px;
  cursor: pointer;
`;

const AlertBadge = styled.div`
  background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
  color: white;
  padding: 8px 16px;
  border-radius: 20px;
  font-size: 12px;
  font-weight: 600;
  margin: 5px 0;
  animation: pulse 2s infinite;
  
  @keyframes pulse {
    0% { opacity: 1; }
    50% { opacity: 0.7; }
    100% { opacity: 1; }
  }
`;

const LoadingSpinner = styled.div`
  border: 4px solid rgba(255, 255, 255, 0.3);
  border-radius: 50%;
  border-top: 4px solid #667eea;
  width: 40px;
  height: 40px;
  animation: spin 1s linear infinite;
  margin: 20px auto;
  
  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

// Color schemes for charts
const COLORS = ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'];
const RISK_COLORS = {
  'AAA': '#4facfe', 'AA': '#00f2fe', 'A': '#a8edea',
  'BBB': '#fed6e3', 'BB': '#f093fb', 'B': '#f5576c',
  'CCC': '#ff9a9e', 'CC': '#fecfef', 'C': '#ff6b6b', 'D': '#ee5a24'
};

export default function CreditDashboard() {
  const [companies, setCompanies] = useState([]);
  const [selectedCompany, setSelectedCompany] = useState(null);
  const [scores, setScores] = useState([]);
  const [latestScore, setLatestScore] = useState(null);
  const [explanations, setExplanations] = useState(null);
  const [sentiment, setSentiment] = useState(null);
  const [dataStatus, setDataStatus] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [filters, setFilters] = useState({
    timeRange: '30d',
    riskCategory: 'all',
    sector: 'all'
  });
  const [alerts, setAlerts] = useState([]);

  const API_BASE = 'http://localhost:8000';

  // Fetch functions
  const fetchCompanies = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/companies`);
      setCompanies(response.data);
      if (response.data.length > 0 && !selectedCompany) {
        setSelectedCompany(response.data[0]);
      }
    } catch (err) {
      setError('Failed to fetch companies');
    }
  }, [selectedCompany]);

  const fetchScores = useCallback(async (companyId) => {
    if (!companyId) return;
    try {
      const response = await axios.get(`${API_BASE}/companies/${companyId}/scores?limit=50`);
      setScores(response.data);
      
      // Get latest score
      if (response.data.length > 0) {
        setLatestScore(response.data[0]);
        
        // Fetch explanations for latest score
        const explResponse = await axios.get(`${API_BASE}/scores/${response.data[0].id}/explanations`);
        setExplanations(explResponse.data);
      }
    } catch (err) {
      console.error('Failed to fetch scores:', err);
    }
  }, []);

  const fetchSentiment = useCallback(async (companyId) => {
    if (!companyId) return;
    try {
      const response = await axios.get(`${API_BASE}/companies/${companyId}/sentiment?days_back=30`);
      setSentiment(response.data);
    } catch (err) {
      console.error('Failed to fetch sentiment:', err);
    }
  }, []);

  const fetchDataStatus = useCallback(async () => {
    try {
      const response = await axios.get(`${API_BASE}/data/status`);
      setDataStatus(response.data);
    } catch (err) {
      console.error('Failed to fetch data status:', err);
    }
  }, []);

  // Initialize data and polling
  useEffect(() => {
    const initializeData = async () => {
      setLoading(true);
      await fetchCompanies();
      await fetchDataStatus();
      setLoading(false);
    };
    
    initializeData();
    
    // Set up polling for real-time updates
    const interval = setInterval(() => {
      fetchDataStatus();
      if (selectedCompany) {
        fetchScores(selectedCompany.id);
        fetchSentiment(selectedCompany.id);
      }
    }, 30000); // Poll every 30 seconds
    
    return () => clearInterval(interval);
  }, [fetchCompanies, fetchDataStatus, fetchScores, fetchSentiment, selectedCompany]);

  // Fetch company-specific data when selection changes
  useEffect(() => {
    if (selectedCompany) {
      fetchScores(selectedCompany.id);
      fetchSentiment(selectedCompany.id);
    }
  }, [selectedCompany, fetchScores, fetchSentiment]);

  // Generate alerts based on score changes
  useEffect(() => {
    if (scores.length >= 2) {
      const latest = scores[0];
      const previous = scores[1];
      const scoreDiff = latest.credit_score - previous.credit_score;
      
      if (Math.abs(scoreDiff) > 50) {
        const alert = {
          id: Date.now(),
          type: scoreDiff > 0 ? 'improvement' : 'decline',
          message: `${selectedCompany?.name} score ${scoreDiff > 0 ? 'improved' : 'declined'} by ${Math.abs(scoreDiff).toFixed(0)} points`,
          timestamp: new Date().toISOString(),
          severity: Math.abs(scoreDiff) > 100 ? 'high' : 'medium'
        };
        
        setAlerts(prev => [alert, ...prev.slice(0, 4)]); // Keep last 5 alerts
      }
    }
  }, [scores, selectedCompany]);

  // Data processing functions
  const getScoreHistory = () => {
    return scores.map(score => ({
      date: new Date(score.score_date).toLocaleDateString(),
      score: score.credit_score,
      confidence: score.confidence * 100,
      risk: score.risk_category
    })).reverse();
  };

  const getFeatureContributions = () => {
    if (!explanations?.feature_contributions) return [];
    
    const positive = explanations.feature_contributions.positive || [];
    const negative = explanations.feature_contributions.negative || [];
    
    return [...positive, ...negative]
      .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
      .slice(0, 10)
      .map(item => ({
        feature: item.description || item.feature_name,
        contribution: item.contribution,
        value: item.value,
        importance: item.importance
      }));
  };

  const getSentimentData = () => {
    if (!sentiment?.latest_articles) return [];
    
    return sentiment.latest_articles.map((article, index) => ({
      id: index,
      title: article.title.substring(0, 50) + '...',
      sentiment: article.sentiment_score,
      date: new Date(article.published_at).toLocaleDateString(),
      source: article.source
    }));
  };

  const getRiskDistribution = () => {
    if (!companies.length) return [];
    
    const distribution = {};
    companies.forEach(company => {
      // Mock risk distribution - in real app, get from latest scores
      const risks = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC'];
      const risk = risks[Math.floor(Math.random() * risks.length)];
      distribution[risk] = (distribution[risk] || 0) + 1;
    });
    
    return Object.entries(distribution).map(([risk, count]) => ({
      risk,
      count,
      color: RISK_COLORS[risk]
    }));
  };

  // Render functions
  const renderOverview = () => (
    <div>
      <GridContainer>
        <MetricCard risk={latestScore?.risk_category === 'AAA' || latestScore?.risk_category === 'AA' ? 'low' : 
                         latestScore?.risk_category === 'A' || latestScore?.risk_category === 'BBB' ? 'medium' : 'high'}>
          <h2>{latestScore?.credit_score?.toFixed(0) || 'N/A'}</h2>
          <p>Credit Score</p>
          <small>{latestScore?.risk_category || 'No Rating'}</small>
        </MetricCard>
        
        <MetricCard>
          <h2>{(latestScore?.confidence * 100)?.toFixed(1) || 'N/A'}%</h2>
          <p>Confidence</p>
          <small>Model Certainty</small>
        </MetricCard>
        
        <MetricCard>
          <h2>{sentiment?.total_articles || 0}</h2>
          <p>News Articles</p>
          <small>Last 30 Days</small>
        </MetricCard>
        
        <MetricCard>
          <h2>{sentiment?.average_sentiment?.toFixed(2) || 'N/A'}</h2>
          <p>Avg Sentiment</p>
          <small>-1 to +1 Scale</small>
        </MetricCard>
      </GridContainer>

      <GridContainer>
        <Card>
          <h3>Score History</h3>
          <ResponsiveContainer width="100%" height={300}>
            <ComposedChart data={getScoreHistory()}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis yAxisId="left" domain={[300, 850]} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 100]} />
              <Tooltip />
              <Legend />
              <Area yAxisId="left" type="monotone" dataKey="score" fill="#667eea" fillOpacity={0.3} />
              <Line yAxisId="left" type="monotone" dataKey="score" stroke="#667eea" strokeWidth={3} />
              <Bar yAxisId="right" dataKey="confidence" fill="#764ba2" opacity={0.6} />
            </ComposedChart>
          </ResponsiveContainer>
        </Card>

        <Card>
          <h3>Feature Contributions</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={getFeatureContributions()} layout="horizontal">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" />
              <YAxis dataKey="feature" type="category" width={120} />
              <Tooltip />
              <Bar dataKey="contribution" fill={(entry) => entry.contribution > 0 ? '#4facfe' : '#f5576c'} />
            </BarChart>
          </ResponsiveContainer>
        </Card>
      </GridContainer>

      {alerts.length > 0 && (
        <Card>
          <h3>Recent Alerts</h3>
          {alerts.map(alert => (
            <AlertBadge key={alert.id}>
              {alert.message} - {new Date(alert.timestamp).toLocaleTimeString()}
            </AlertBadge>
          ))}
        </Card>
      )}
    </div>
  );

  const renderAnalytics = () => (
    <GridContainer>
      <Card>
        <h3>Sentiment Analysis</h3>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterPlot data={getSentimentData()}>
            <CartesianGrid />
            <XAxis dataKey="date" />
            <YAxis dataKey="sentiment" domain={[-1, 1]} />
            <Tooltip />
            <Scatter dataKey="sentiment" fill="#667eea" />
          </ScatterPlot>
        </ResponsiveContainer>
      </Card>

      <Card>
        <h3>Risk Distribution</h3>
        <ResponsiveContainer width="100%" height={300}>
          <PieChart>
            <Pie
              data={getRiskDistribution()}
              cx="50%"
              cy="50%"
              innerRadius={60}
              outerRadius={120}
              dataKey="count"
              nameKey="risk"
            >
              {getRiskDistribution().map((entry, index) => (
                <Cell key={`cell-${index}`} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip />
            <Legend />
          </PieChart>
        </ResponsiveContainer>
      </Card>
    </GridContainer>
  );

  const renderExplanations = () => (
    <Card>
      <h3>Score Explanation</h3>
      {explanations ? (
        <div>
          <p><strong>Summary:</strong> {explanations.summary}</p>
          
          <h4>Key Factors:</h4>
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div>
              <h5>Positive Contributors</h5>
              {explanations.feature_contributions?.positive?.slice(0, 5).map((item, index) => (
                <div key={index} style={{ padding: '8px', background: '#e8f5e8', margin: '5px 0', borderRadius: '5px' }}>
                  <strong>{item.description}</strong>: +{item.contribution.toFixed(1)}
                </div>
              ))}
            </div>
            
            <div>
              <h5>Negative Contributors</h5>
              {explanations.feature_contributions?.negative?.slice(0, 5).map((item, index) => (
                <div key={index} style={{ padding: '8px', background: '#ffe8e8', margin: '5px 0', borderRadius: '5px' }}>
                  <strong>{item.description}</strong>: {item.contribution.toFixed(1)}
                </div>
              ))}
            </div>
          </div>

          <h4>Recommendations:</h4>
          <ul>
            {explanations.recommendations?.map((rec, index) => (
              <li key={index}>{rec}</li>
            ))}
          </ul>
        </div>
      ) : (
        <p>No explanations available. Generate a score first.</p>
      )}
    </Card>
  );

  if (loading) {
    return (
      <DashboardContainer>
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', width: '100%' }}>
          <LoadingSpinner />
        </div>
      </DashboardContainer>
    );
  }

  return (
    <DashboardContainer>
      <Sidebar>
        <h2 style={{ color: '#333', marginBottom: '20px' }}>Credit Intelligence</h2>
        
        <FilterContainer style={{ flexDirection: 'column', gap: '10px' }}>
          <FilterSelect 
            value={filters.timeRange} 
            onChange={(e) => setFilters({...filters, timeRange: e.target.value})}
          >
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
            <option value="1y">Last Year</option>
          </FilterSelect>
          
          <FilterSelect 
            value={filters.riskCategory} 
            onChange={(e) => setFilters({...filters, riskCategory: e.target.value})}
          >
            <option value="all">All Risk Levels</option>
            <option value="investment">Investment Grade</option>
            <option value="speculative">Speculative Grade</option>
          </FilterSelect>
        </FilterContainer>

        <h3 style={{ color: '#333', marginTop: '20px' }}>Companies</h3>
        <CompanyList>
          {companies.map(company => (
            <CompanyItem
              key={company.id}
              selected={selectedCompany?.id === company.id}
              onClick={() => setSelectedCompany(company)}
            >
              <strong>{company.ticker}</strong>
              <br />
              <small>{company.name}</small>
            </CompanyItem>
          ))}
        </CompanyList>

        {dataStatus && (
          <Card style={{ marginTop: '20px', padding: '15px' }}>
            <h4>System Status</h4>
            <p><strong>Companies:</strong> {dataStatus.total_companies}</p>
            <p><strong>Last Update:</strong> {new Date(dataStatus.last_updated).toLocaleTimeString()}</p>
          </Card>
        )}
      </Sidebar>

      <MainContent>
        <Header>
          <h1>{selectedCompany?.name || 'Select a Company'}</h1>
          <p>{selectedCompany?.ticker}</p>
        </Header>

        <TabContainer>
          <Tab active={activeTab === 'overview'} onClick={() => setActiveTab('overview')}>
            Overview
          </Tab>
          <Tab active={activeTab === 'analytics'} onClick={() => setActiveTab('analytics')}>
            Analytics
          </Tab>
          <Tab active={activeTab === 'explanations'} onClick={() => setActiveTab('explanations')}>
            Explanations
          </Tab>
        </TabContainer>

        {error && (
          <Card style={{ background: '#ffe8e8', border: '1px solid #ffcccc' }}>
            <p style={{ color: '#cc0000' }}>{error}</p>
          </Card>
        )}

        {activeTab === 'overview' && renderOverview()}
        {activeTab === 'analytics' && renderAnalytics()}
        {activeTab === 'explanations' && renderExplanations()}
      </MainContent>
    </DashboardContainer>
  );
}
