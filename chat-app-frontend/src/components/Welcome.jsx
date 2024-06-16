import React from 'react'
import {useUser} from "../context/UserContext";

const Welcome = () => {
    const { currentUser } = useUser();
    return(
        <div className='welcome'>
            <img src={"https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExeDU0ZDN2amxvY3VnMjBiMHFsdnNvOGhyemYwdTViaGp2NXhoaHk5aiZlcD12MV9pbnRlcm5hbF9naWZfYnlfaWQmY3Q9cw/KGMzZvWa5su2O5LCVR/giphy.gif"} alt="" />
            <h1>
                Welcome, <span>{currentUser?.username}!</span>
            </h1>
            <h3>Please select a chat  to start messaging.</h3>
        </div>
    )
}

export default Welcome