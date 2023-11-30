import React from 'react'
import Robot from "../assets/robot.gif";
import {useUser} from "../context/UserContext";

const Welcome = () => {
    const { currentUser } = useUser();
    return(
        <div className='welcome'>
            <img src={Robot} alt="" />
            <h1>
                Welcome, <span>{currentUser?.username}!</span>
            </h1>
            <h3>Please select a chat  to start messaging.</h3>
        </div>
    )
}

export default Welcome