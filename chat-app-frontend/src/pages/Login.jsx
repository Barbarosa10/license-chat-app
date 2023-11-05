import React from "react";
import {motion} from "framer-motion"


const Login = () => {
    return(
        <div className="formContainer">
            <motion.div initial={{ opacity: 0 }} animate={{rotate: 360, opacity: 1 }} transition={{ duration: 1}} className="formWrapper">
                <span className="logo">Sentiment Analysis Chat</span>
                <span className="title">Login</span>
                <form action="home">
                    <input type="email" placeholder="Email"/>
                    <input type="password" placeholder="Password"/>
                    <button>Sign in</button>
                </form>
                <div className="linkParagraph">
                    <p>Forgot password? <a href="forgotPassword">Change Password</a></p>
                    <p>You don't have an account? <a href="register">Register</a></p>
                </div>

            </motion.div>
        </div>
    )
}

export default Login