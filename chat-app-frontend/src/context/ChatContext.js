import {
    createContext,
    useContext,
    useReducer,
    useState
  } from "react";
  
  export const ChatContext = createContext();
  
  export const ChatContextProvider = ({ children }) => {
    const INITIAL_STATE = {
      chatId: "null",
      user: {},
    };
    const [chatSelected, setChatSelected] = useState(false);
  
    const chatReducer = (state, action) => {
      switch (action.type) {
        case "CHANGE_USER":
          console.log(action.user);
          return {
            user: action.user,
            chatId: action.user.id,
          };
  
        default:
          return state;
      }
    };
    const selectChat = () => {
        setChatSelected(true);
    }
  
    const [state, dispatch] = useReducer(chatReducer, INITIAL_STATE);
  
    return (
      <ChatContext.Provider value={{chatSelected, selectChat, data:state, dispatch }}>
        {children}
      </ChatContext.Provider>
    );
  };

  export const useChat = () => {
    const context = useContext(ChatContext);
    if (!context) {
      throw new Error('useChat must be used within a ChatContextProvider');
    }
    return context;
  };
  