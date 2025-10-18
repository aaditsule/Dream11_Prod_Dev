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
            if (err.response) {
                // The request was made and the server responded with a status code that falls out of the range of 2xx
                setError(err.response.data.detail || 'An error occurred on the server.');
            } else if (err.request) {
                // The request was made but no response was received
                setError('Could not connect to the server. Please check your connection.');
            } else {
                // Something happened in setting up the request that triggered an Error
                setError('An unexpected error occurred.');
            }
        } finally {
            setIsLoading(false);
        }
    };

    // --- RATIONALE MODAL ---
    const RationaleModal = ({ player, onClose }) => {
        const rationale = player.rationale || {};

        // Sort features by the absolute magnitude of their impact
        const sortedFeatures = Object.entries(rationale).sort(([, a], [, b]) => Math.abs(b || 0) - Math.abs(a || 0));

        // Find the maximum absolute value for better bar scaling
        const maxAbsValue = Math.max(...sortedFeatures.map(([, v]) => Math.abs(v || 0)));

        // Helper to make feature names readable
        const formatFeatureName = (name) => {
            return name
                .replace('role_', 'Role ')
                .replace('avg_fp_', 'Avg FP ')
                .replace('_', ' ')
                .replace(/\b\w/g, l => l.toUpperCase()); // Capitalize each word
        };

        return (
            <div className="modal-backdrop" onClick={onClose}>
                <div className="modal-content" onClick={e => e.stopPropagation()}>
                    <h3>Rationale for {player.name}</h3>
                    <p>Predicted Points: {(player.predicted_fp || 0).toFixed(2)}</p>
                    <ul className="rationale-list">
                        {sortedFeatures.map(([feature, value]) => {
                            const val = value || 0;
                            const width = Math.abs(val) < 5 ? (Math.abs(val) / 5) * 100 : (Math.abs(val) / maxAbsValue) * 100;
                            const isBarFeature = !['avg_fp_last_5', 'matches_played'].includes(feature);
                            return (
                                <li key={feature} className={val >= 0 ? 'positive' : 'negative'}>
                                    <span className="feature-name">{formatFeatureName(feature)}</span>
                                    {isBarFeature ? (
                                        <>
                                            <div className="bar-container">
                                                <div className="bar" style={{ width: `${width}%`, backgroundColor: val >= 0 ? '#4CAF50' : '#F44336' }}></div>
                                            </div>
                                            <span className="feature-value">{val.toFixed(2)}</span>
                                        </>
                                    ) : (
                                        // Render only the value for specified features
                                        <span className="feature-value-no-bar">{val.toFixed(2)}</span>
                                    )}
                                </li>
                            );
                        })}
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
                            {teamData.recommended_xi.sort((a, b) => b.predicted_fp - a.predicted_fp).map((player) => (
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
                            <div className="summary-item"><span>WK</span><strong>{teamData.summary.role_counts['WK'] || 0}</strong></div>
                            <div className="summary-item"><span>BAT</span><strong>{teamData.summary.role_counts['BAT'] || 0}</strong></div>
                            <div className="summary-item"><span>AR</span><strong>{teamData.summary.role_counts['AR'] || 0}</strong></div>
                            <div className="summary-item"><span>BOWL</span><strong>{teamData.summary.role_counts['BOWL'] || 0}</strong></div>
                        <h3>Team Count</h3>
                        {Object.entries(teamData.summary.team_counts || {}).map(([team, count]) => (
                             <div key={team} className="summary-item"><span>{team}</span><strong>{count}</strong></div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}

export default App;