import React from 'react'
import Robot from "../assets/robot.gif";
import {useUser} from "../context/UserContext";

const Welcome = () => {
    const { currentUser } = useUser();
    return(
        <div className='welcome'>
            <img src={"https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDU0ZDN2amxvY3VnMjBiMHFsdnNvOGhyemYwdTViaGp2NXhoaHk5aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/KGMzZvWa5su2O5LCVR/giphy.gif"} alt="" />
            {/* <iframe src="https://giphy.com/embed/KGMzZvWa5su2O5LCVR" width="480" height="461" frameBorder="0" class="giphy-embed"></iframe> */}
            <h1>
                Welcome, <span>{currentUser?.username}!</span>
            </h1>
            <h3>Please select a chat  to start messaging.</h3>
        </div>
    )
}

export default Welcome