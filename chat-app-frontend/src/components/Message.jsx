import React from 'react'

const Message = () => {
    return(
        <div className='message'>
            <div className='messageInfo'>
                <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" />
                <span>Just now</span>
            </div>
            <div className='messageContent'>
                <p>Hello bro</p>
                {/* <img src="https://images.pexels.com/photos/220453/pexels-photo-220453.jpeg?auto=compress&cs=tinysrgb&dpr=1&w=500" alt="" /> */}
            </div>
        </div>        
    )
}

export default Message