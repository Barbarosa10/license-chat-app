import React from "react";
import Add from "../images/addAvatar.png"
import {motion} from "framer-motion"


const Register = () => {
    return(
        <div className="formContainer">
            <motion.div  initial={{ opacity: 0 }} animate={{rotate: 360, opacity: 1 }} transition={{ duration: 1}} className="formWrapper">
                <span className="logo">Sentiment Analysis Chat</span>
                <span className="title">Register</span>
                <form action="">
                    <input type="text" placeholder="Username"/>
                    <input type="email" placeholder="Email"/>
                    <input type="password" placeholder="Password"/>
                    <input style={{display: "none"}} type="file" id="file" />
                    <label htmlFor="file">
                        <img src={Add} alt="" />
                        <span>Add an avatar</span>
                    </label>
                    <button>Sign up</button>
                </form>
                <div className="linkParagraph">
                    <p>Already registered? <a href="login">Login</a></p>
                </div>
            </motion.div>
        </div>
    )
}

export default Register