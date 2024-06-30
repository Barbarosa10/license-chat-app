import React, { createContext, useContext, useState, useRef } from 'react';

const SocketContext = createContext();

export const SocketProvider = ({ children }) => {
  let [socket, setSocket] = useState(null);
  const [socketVar, setSocketVar] = useState(1);

  const setSocketAtInitialization = (sockett) => {
    setSocket(sockett);
    setSocketVar(-socketVar);
  };

  return (
    <SocketContext.Provider value={{ socketv: socket, socketVar, setSocket: setSocketAtInitialization }}>
      {children}
    </SocketContext.Provider>
  );
};

export const useSocket = () => {
  const context = useContext(SocketContext);
  if (!context) {
    throw new Error('useMessages must be used within a MessageProvider');
  }
  return context;
};