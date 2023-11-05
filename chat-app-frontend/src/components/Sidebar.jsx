import React from 'react'
import Navbar from "./Navbar";
import Searchbar from './Searchbar';
import Conversations from './Conversations';

const Sidebar = () => {
    return(
        <div className='sidebar'>
            <Navbar/>
            <Searchbar/>
            <Conversations/>
        </div>
    )
}

export default Sidebar