import React, { useState, useEffect} from "react";
import { useNavigate, Link } from "react-router-dom";
import Add from "../images/addAvatar.png"
import {motion} from "framer-motion"
import axios from '../utils/axiosConfig';
import { registerRoute } from "../utils/ApiRoute";
import {useUser} from "../context/UserContext";

export default function Register (){
    const { setCurrentUser } = useUser();
    const [loading, setLoading] = useState(false);
    const navigate = useNavigate();
    const [selectedImage, setSelectedImage] = useState(null);
    const [msg, setMsg] = useState(null);
    
    useEffect(() => {
        if (localStorage.getItem(process.env.REACT_APP_LOCALHOST_KEY)) {
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
      
          reader.onload = function (event) {
            const img = new Image();
            img.onload = function () {
              const canvas = document.createElement('canvas');
              const ctx = canvas.getContext('2d');
      
              const scaleFactor = 48 / Math.max(img.width, img.height);
              canvas.width = img.width * scaleFactor;
              canvas.height = img.height * scaleFactor;
      
              ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      
              const scaledBase64 = canvas.toDataURL('image/jpeg', 0.8); // Adjust quality as needed (0.8 is 80% quality)
              const contentBase64 = scaledBase64.split(',')[1];
              setSelectedImage(contentBase64);
            };
            img.src = event.target.result;
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
                if(response){
                    let data = response.data;
    
                    if(data.status === false){
                        setMsg(data.msg);
                    } else if(data.status === true){
                        setCurrentUser(data.user);
                        localStorage.setItem(
                            process.env.REACT_APP_LOCALHOST_KEY,
                            JSON.stringify(data)
                        );
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