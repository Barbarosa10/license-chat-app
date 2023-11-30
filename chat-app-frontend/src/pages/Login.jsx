import React, { useState, useEffect} from "react";
import {motion} from "framer-motion"
import { loginRoute } from "../utils/ApiRoute";
import axios from "axios"
import { useNavigate} from "react-router-dom";
import {useUser} from "../context/UserContext";

const localhost_key = "chat-app-current-user"



const Login = () => {
    const { currentUser, setCurrentUser } = useUser();
    const navigate = useNavigate();

    useEffect(() => {
        if (localStorage.getItem(localhost_key)) {
        // if(currentUser){
          navigate("/");
        }
      }, []);

    const [msg, setMsg] = useState(null);
    
    useEffect(() => {
        return () => {
          setMsg(null);
        };
      }, []);

    const handleValidation = (username, password) => {
        if (username.length < 3) {
            setMsg("Username should be greater than 3 characters!")
            return false;
        } else if (password.length < 8) {
            setMsg("Password should be equal or greater than 8 characters!")
            return false;
        }
        return true;
    }
      
    const handleSubmit = async (e) => {
        e.preventDefault();

        const username = e.target[0].value;
        const password = e.target[1].value;

        if(handleValidation(username, password)){
            try{
                const response = await axios.post(loginRoute, {
                    username, 
                    password
                }
                );
                console.log(response);

                if(response){
                    let data = response.data;
    
                    if(data.status === false){
                        setMsg(data.msg);
                    } else if(data.status === true){
                        setCurrentUser(data.user);
                        localStorage.setItem(
                            localhost_key,
                            JSON.stringify(data)
                        );
                        navigate("/");
                    }
                }
            }catch(error){
                console.error(error);
            }


        }
    }

    return(
        <div className="formContainer">
            <motion.div initial={{ opacity: 0 }} animate={{rotate: 360, opacity: 1 }} transition={{ duration: 1}} className="formWrapper">
                <span className="logo">Sentiment Analysis Chat</span>
                <span className="title">Login</span>
                <form onSubmit={handleSubmit}>
                    <input type="username" placeholder="Username"/>
                    <input type="password" placeholder="Password"/>
                    <button>Sign in</button>
  
                </form>
                <div className="responseMessage">
                        {msg}
                    </div>
                <div className="linkParagraph">
                    <p>Forgot password? <a href="forgotPassword">Change Password</a></p>
                    <p>You don't have an account? <a href="register">Register</a></p>
                </div>

            </motion.div>
        </div>
    )
}

export default Login