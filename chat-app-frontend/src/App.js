import Register from "./pages/Register";
import Login from "./pages/Login";
import Home from "./pages/Home";
import ForgotPassword from "./pages/ForgotPassword";
import ChangePassword from "./pages/ChangePassword";
import { UserProvider } from './context/UserContext';
import {ConversationProvider} from "./context/ConversationContext";
import {ChatContextProvider} from "./context/ChatContext";
import {ToastContainer} from "react-toastify"

import "./style.scss"

import {BrowserRouter, Routes, Route, Navigate} from "react-router-dom";

function App() {
  return (

          <BrowserRouter>
            <UserProvider>
              <ConversationProvider>
                <ChatContextProvider>
                  <Routes>
                      <Route path="" element={<Navigate to="/home"/>}/>
                      <Route path="Home" element={<Home/>}/>
                      <Route path="Login" element={<Login/>}/>
                      <Route path="Register" element={<Register/>}/>
                      <Route path="ForgotPassword" element={<ForgotPassword/>}/>
                      <Route path="ChangePassword" element={<ChangePassword/>}/>

                  </Routes>
                </ChatContextProvider>
              </ConversationProvider>
            </UserProvider>
          </BrowserRouter>

  );
}

export default App;
