import React, { createContext, useState, useContext } from 'react';

const PopupContext = createContext();

export const PopupProvider = ({ children }) => {
  const [message, setMessage] = useState('');
  const [showPopup, setShowPopup] = useState(false);

  const showMessage = (msg) => {
    setMessage(msg);
    setShowPopup(true);
    setTimeout(() => {
        hideMessage();
      }, 3000);
  };

  const hideMessage = () => {
    setShowPopup(false);
  };

  return (
    <PopupContext.Provider value={{ message, showMessage, hideMessage, showPopup }}>
      {children}
    </PopupContext.Provider>
  );
};

export const usePopup = () => {
  const context = useContext(PopupContext);
  if (!context) {
    throw new Error('usePopup must be used within a PopupProvider');
  }
  return context;
};