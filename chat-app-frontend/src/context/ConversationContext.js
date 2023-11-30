import React, { createContext, useContext, useState } from 'react';
import Conversations from '../components/Conversations';

const ConversationContext = createContext();

export const ConversationProvider = ({ children }) => {
  const [conversations, setConversations] = useState([]);

  const addConversation = (conversation) => {
    setConversations((prevConversations) => [...prevConversations, conversation]);
  };
  
  const setConversationsAtInitialization = (conversations) => {
    setConversations(conversations);
  };

  return (
    <ConversationContext.Provider value={{ conversations, addConversation, setConversationsAtInitialization }}>
      {children}
    </ConversationContext.Provider>
  );
};

export const useConversation = () => {
  const context = useContext(ConversationContext);
  if (!context) {
    throw new Error('useConversation must be used within a ConversationProvider');
  }
  return context;
};