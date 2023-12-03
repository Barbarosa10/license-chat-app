import React from 'react';

const CallPopup = ({ onAnswer, onDecline }) => {
    
    return (
        <div className="call-popup">
        <p>Incoming Call...</p>
        <div className="button-container">
            <button onClick={onAnswer}>Answer</button>
            <button onClick={onDecline}>Decline</button>
        </div>
        </div>
    );
};

export default CallPopup;