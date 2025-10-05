import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const API_URL = "http://localhost:8000/api/predict_team/";

function App() {
    const [file, setFile] = useState(null);
    const [teamData, setTeamData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState('');
    const [activeRationale, setActiveRationale] = useState(null);

    const handleFileChange = (event) => {
        setTeamData(null);
        setError('');
        setFile(event.target.files[0]);
    };

    const handleUpload = async () => {
        if (!file) {
            setError('Please select a match file first.');
            return;
        }
        setIsLoading(true);
        setError('');
        setTeamData(null);

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await axios.post(API_URL, formData, {
                headers: { 'Content-Type': 'multipart/form-data' },
            });
            setTeamData(response.data);
        } catch (err) {
            setError(err.response?.data?.detail || 'An error occurred. Please try again.');
        } finally {
            setIsLoading(false);
        }
    };
    
    // This is the component with the fix
    const RationaleModal = ({ player, onClose }) => {
        const rationale = player.rationale || {};
        const sortedFeatures = Object.entries(rationale).sort(([,a],[,b]) => Math.abs(b || 0) - Math.abs(a || 0));
    
        return (
            <div className="modal-backdrop" onClick={onClose}>
                <div className="modal-content" onClick={e => e.stopPropagation()}>
                    <h3>Rationale for {player.name}</h3>
                    <p>Predicted Points: {(player.predicted_fp || 0).toFixed(2)}</p>
                    <ul>
                        {sortedFeatures.map(([feature, value]) => (
                            <li key={feature} className={(value || 0) > 0 ? 'positive' : 'negative'}>
                                <strong>{feature.replace(/role_|avg_fp_/g, '').replace('_', ' ')}:</strong> 
                                {/* Use (value || 0) to prevent calling .toFixed() on null */}
                                <span>{(value || 0).toFixed(2)}</span>
                                <div className="bar-container">
                                    <div className="bar" style={{ width: `${Math.abs(value || 0) * 5}%`, background: (value || 0) > 0 ? '#4CAF50' : '#F44336' }}></div>
                                </div>
                            </li>
                        ))}
                    </ul>
                    <button onClick={onClose}>Close</button>
                </div>
            </div>
        );
    };

    return (
        <div className="container">
            {activeRationale && <RationaleModal player={activeRationale} onClose={() => setActiveRationale(null)} />}
            <header>
                <h1>Dream Team Predictor</h1>
                <p>Upload a match JSON file to generate the optimal fantasy XI.</p>
            </header>

            <div className="upload-section">
                <input type="file" onChange={handleFileChange} accept=".json" />
                <button onClick={handleUpload} disabled={isLoading}>
                    {isLoading ? 'Generating...' : 'Generate Team'}
                </button>
            </div>

            {error && <p className="error-message">{error}</p>}
            
            {teamData && (
                <div className="results-grid">
                    <div className="team-display">
                        <h2>Recommended XI</h2>
                        <div className="player-cards-container">
                            {teamData.recommended_xi.map((player) => (
                                <div key={player.player_id} className="player-card">
                                    <h4>{player.name}</h4>
                                    <p>{player.role} | {player.team}</p>
                                    <p>Credits: <strong>{(player.credits || 0).toFixed(2)}</strong></p>
                                    <p>Predicted FP: <strong>{(player.predicted_fp || 0).toFixed(2)}</strong></p>
                                    <button className="rationale-btn" onClick={() => setActiveRationale(player)}>Show Rationale</button>
                                </div>
                            ))}
                        </div>
                    </div>
                    <div className="summary-panel">
                        <h2>Summary & Rules</h2>
                        <div className="summary-item">
                            <span>Total Points</span>
                            <strong>{(teamData.summary.total_predicted_points || 0).toFixed(2)}</strong>
                        </div>
                        <div className="summary-item">
                            <span>Credits Used</span>
                            <strong>{(teamData.summary.total_credits_used || 0).toFixed(2)} / 100</strong>
                        </div>
                        <h3>Role Count</h3>
                        {Object.entries(teamData.summary.role_counts).map(([role, count]) => (
                             <div key={role} className="summary-item"><span>{role}</span><strong>{count}</strong></div>
                        ))}
                        <h3>Team Count</h3>
                        {Object.entries(teamData.summary.team_counts).map(([team, count]) => (
                             <div key={team} className="summary-item"><span>{team}</span><strong>{count}</strong></div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;