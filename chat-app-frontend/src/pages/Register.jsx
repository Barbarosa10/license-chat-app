import React, { useState, useEffect} from "react";
import { useNavigate, Link } from "react-router-dom";
import Add from "../images/addAvatar.png"
import {motion} from "framer-motion"
import axios from "axios"
import { registerRoute } from "../utils/ApiRoute";
import {useUser} from "../context/UserContext";
const localhost_key = "chat-app-current-user";

export default function Register (){
    const { currentUser, setCurrentUser } = useUser();
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const [selectedImage, setSelectedImage] = useState(null);
    const [msg, setMsg] = useState(null);
    
    useEffect(() => {
        if (localStorage.getItem(localhost_key)) {
          navigate("/");
        }
      }, []);

    useEffect(() => {
        return () => {
          setMsg(null);
        };
      }, []);

    const handleImageChange = (e) => {
        const file = e.target.files[0];
    
        if (file) {
          const reader = new FileReader();
    
          reader.onload = function (e) {
            const contentBase64 = e.target.result.split(',')[1];
            setSelectedImage(contentBase64);


          };
    
          reader.readAsDataURL(file);
        }
      };

    const handleValidation = (username, email, password) => {
        if (username.length < 3) {
            setMsg("Username should be greater than 3 characters!")
            return false;
        } else if (password.length < 8) {
            setMsg("Password should be equal or greater than 8 characters!")
            return false;
        } else if (email === "") {
            setMsg("Email is required!");
            return false;
        } else if(selectedImage === null){
            setMsg("Avatar image is required!");
            return false;
        }
        return true;
    }
      
    const handleSubmit = async (e) => {
        setLoading(true);
        e.preventDefault();

        const username = e.target[0].value;
        const email = e.target[1].value;
        const password = e.target[2].value;

        if(handleValidation(username, email, password)){
            try{
                const response = await axios.post(registerRoute, {
                    username, 
                    email, 
                    password, 
                    avatarImage : selectedImage
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
                        console.log(data.user);
                        console.log(currentUser);
                        navigate("/");
                    }
                }
            }catch(error){
                console.error(error);
            }


        }

        setLoading(false);
    }

    return(
        <>
            <div className="formContainer">
                <motion.div  initial={{ opacity: 0 }} animate={{rotate: 360, opacity: 1 }} transition={{ duration: 1}} className="formWrapper">
                    <span className="logo">Sentiment Analysis Chat</span>
                    <span className="title">Register</span>
                    <form onSubmit={handleSubmit}>
                        <input type="text" placeholder="Username"/>
                        <input type="email" placeholder="Email"/>
                        <input type="password" placeholder="Password"/>
                        <input style={{display: "none"}} type="file" id="file" onChange={handleImageChange} />
                        <label htmlFor="file">
                            <img src={Add} alt="" />
                            <span>Add an avatar</span>
                        </label>
                        <button disabled={loading}>Sign up</button>

                        
                        
                        
                    </form>
                    <div className="responseMessage">
                            {loading===false && msg}
                            {loading && "Uploading and compressing the image please wait..."}
                        </div>

                    <div className="linkParagraph">
                        <p>Already registered? <Link to="/login">Login</Link></p>

                    </div>
                </motion.div>
            </div>



        </>
    )
}

