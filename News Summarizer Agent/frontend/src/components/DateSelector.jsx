import React from "react";
import DatePicker from "react-datepicker";
import "react-datepicker/dist/react-datepicker.css";

const DateSelector = ({ selectedDate, setSelectedDate, onDateChange }) => {
  return (
    <DatePicker
      selected={selectedDate}
      onChange={(date) => setSelectedDate(date)}
      onCalendarClose={() => onDateChange(selectedDate)} // only fetch when date selection is complete
      maxDate={new Date()}
      dateFormat="yyyy-MM-dd"
    />
  );
};

export default DateSelector;
