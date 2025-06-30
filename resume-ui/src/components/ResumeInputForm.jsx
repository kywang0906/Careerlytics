import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

const ResumeInputForm = () => {
  const navigate = useNavigate();
  // Form state
  const [about, setAbout] = useState('');
  const [educations, setEducations] = useState([{ school: '', major: '', startYear: '', endYear: '' }]);
  const [experiences, setExperiences] = useState([{ company: '', title: '', description: '', startYear: '', endYear: '' }]);
  const [projects, setProjects] = useState([{ name: '', description: '' }]);
  const [publications, setPublications] = useState([{ name: '', description: '' }]);
  const [courses, setCourses] = useState([{ name: '' }]);
  const [certifications, setCertifications] = useState([{ name: '', description: '' }]);
  const [loading, setLoading] = useState(false);

  // Handlers for adding/removing entries
  const addEntry = (list, setList, template) => setList([...list, { ...template }]);
  const removeEntry = (list, setList) => list.length > 1 && setList(list.slice(0, -1));

  // Handlers for field changes
  const handleChange = (list, setList, index, field, value) => {
    const updated = [...list];
    updated[index][field] = value;
    setList(updated);
  };

  const handleSubmit = async () => {
    setLoading(true);

    const payload = {
      about,
      experience: experiences.map(e => ({
        company: e.company,
        title: e.title,
        description: e.description,
        start_year: e.startYear,
        end_year: e.endYear
      })),
      education: educations.map(e => ({
        school: e.school,
        major: e.major,
        start_year: e.startYear,
        end_year: e.endYear
      })),
      projects: projects.map(p => ({ name: p.name, description: p.description })),
      publications: publications.map(p => ({ name: p.name, description: p.description })),
      courses: courses.map(c => c.name),
      certifications: certifications.map(c => c.name)
    };

    try {
      console.log('→ POST', '/predict');
      const res = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const text = await res.text();
        console.error('Predict API failed:', res.status, text);
        alert(`Predict error: ${res.status}`);
        setLoading(false);
        return;
      }

      const { label, score } = await res.json();
      console.log('← predict result', { label, score });

      navigate('/analysis', {
        state: { label, score, payload }
      });
    } catch (err) {
      console.error(err);
      alert('Submission failed.');
    } finally {
      setLoading(false);
    }
  };

  const sectionWrapperStyle = {
    width: '100%',
    maxWidth: '600px',
    margin: '0 auto',
    padding: '2rem'
  };

  return (
    <div style={sectionWrapperStyle}>
      <h2 className="text-center mb-4">Step 1: Enter Resume Information</h2>

      {/* About */}
      <div className="mb-4">
        <label className="form-label">About Me</label>
        <textarea
          className="form-control"
          rows={4}
          placeholder="Brief self-introduction"
          value={about}
          onChange={e => setAbout(e.target.value)}
        />
      </div>

      {/* Education */}
      <div className="mb-4">
        <div className="d-flex align-items-center mb-2">
          <label className="form-label mb-0 flex-grow-1">Education</label>
          <button
            className="btn btn-sm btn-outline-primary ms-2"
            onClick={() => addEntry(educations, setEducations, { school: '', major: '', startYear: '', endYear: '' })}
          >+ Add</button>
          <button
            className="btn btn-sm btn-outline-danger ms-2"
            onClick={() => removeEntry(educations, setEducations)}
            disabled={educations.length <= 1}
          >- Remove</button>
        </div>
        {educations.map((e, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <input
              className="form-control mb-2"
              placeholder="School"
              value={e.school}
              onChange={ev => handleChange(educations, setEducations, i, 'school', ev.target.value)}
            />
            <input
              className="form-control mb-2"
              placeholder="Major"
              value={e.major}
              onChange={ev => handleChange(educations, setEducations, i, 'major', ev.target.value)}
            />
            <div className="d-flex gap-2">
              <input
                className="form-control"
                placeholder="Start Year"
                value={e.startYear}
                onChange={ev => handleChange(educations, setEducations, i, 'startYear', ev.target.value)}
              />
              <input
                className="form-control"
                placeholder="End Year"
                value={e.endYear}
                onChange={ev => handleChange(educations, setEducations, i, 'endYear', ev.target.value)}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Work Experience */}
      <div className="mb-4">
        <div className="d-flex align-items-center mb-2">
          <label className="form-label mb-0 flex-grow-1">Work Experience</label>
          <button
            className="btn btn-sm btn-outline-primary ms-2"
            onClick={() => addEntry(experiences, setExperiences, { company: '', title: '', description: '', startYear: '', endYear: '' })}
          >+ Add</button>
          <button
            className="btn btn-sm btn-outline-danger ms-2"
            onClick={() => removeEntry(experiences, setExperiences)}
            disabled={experiences.length <= 1}
          >- Remove</button>
        </div>
        {experiences.map((exp, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <input
              className="form-control mb-2"
              placeholder="Company"
              value={exp.company}
              onChange={ev => handleChange(experiences, setExperiences, i, 'company', ev.target.value)}
            />
            <input
              className="form-control mb-2"
              placeholder="Title"
              value={exp.title}
              onChange={ev => handleChange(experiences, setExperiences, i, 'title', ev.target.value)}
            />
            <textarea
              className="form-control mb-2"
              rows={3}
              placeholder="Description"
              value={exp.description}
              onChange={ev => handleChange(experiences, setExperiences, i, 'description', ev.target.value)}
            />
            <div className="d-flex gap-2">
              <input
                className="form-control"
                placeholder="Start Year"
                value={exp.startYear}
                onChange={ev => handleChange(experiences, setExperiences, i, 'startYear', ev.target.value)}
              />
              <input
                className="form-control"
                placeholder="End Year"
                value={exp.endYear}
                onChange={ev => handleChange(experiences, setExperiences, i, 'endYear', ev.target.value)}
              />
            </div>
          </div>
        ))}
      </div>

      {/* Projects */}
      <div className="mb-4">
        <div className="d-flex align-items-center mb-2">
          <label className="form-label mb-0 flex-grow-1">Projects</label>
          <button
            className="btn btn-sm btn-outline-primary ms-2"
            onClick={() => addEntry(projects, setProjects, { name: '', description: '' })}
          >+ Add</button>
          <button
            className="btn btn-sm btn-outline-danger ms-2"
            onClick={() => removeEntry(projects, setProjects)}
            disabled={projects.length <= 1}
          >- Remove</button>
        </div>
        {projects.map((p, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <input
              className="form-control mb-2"
              placeholder="Project Name"
              value={p.name}
              onChange={ev => handleChange(projects, setProjects, i, 'name', ev.target.value)}
            />
            <textarea
              className="form-control"
              rows={3}
              placeholder="Description"
              value={p.description}
              onChange={ev => handleChange(projects, setProjects, i, 'description', ev.target.value)}
            />
          </div>
        ))}
      </div>

      {/* Publications */}
      <div className="mb-4">
        <div className="d-flex align-items-center mb-2">
          <label className="form-label mb-0 flex-grow-1">Publications</label>
          <button
            className="btn btn-sm btn-outline-primary ms-2"
            onClick={() => addEntry(publications, setPublications, { name: '', description: '' })}
          >+ Add</button>
          <button
            className="btn btn-sm btn-outline-danger ms-2"
            onClick={() => removeEntry(publications, setPublications)}
            disabled={publications.length <= 1}
          >- Remove</button>
        </div>
        {publications.map((p, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <input
              className="form-control mb-2"
              placeholder="Publication Name"
              value={p.name}
              onChange={ev => handleChange(publications, setPublications, i, 'name', ev.target.value)}
            />
            <textarea
              className="form-control"
              rows={3}
              placeholder="Description"
              value={p.description}
              onChange={ev => handleChange(publications, setPublications, i, 'description', ev.target.value)}
            />
          </div>
        ))}
      </div>

      {/* Courses */}
      <div className="mb-4">
        <div className="d-flex align-items-center mb-2">
          <label className="form-label mb-0 flex-grow-1">Courses</label>
          <button
            className="btn btn-sm btn-outline-primary ms-2"
            onClick={() => addEntry(courses, setCourses, { name: '' })}
          >+ Add</button>
          <button
            className="btn btn-sm btn-outline-danger ms-2"
            onClick={() => removeEntry(courses, setCourses)}
            disabled={courses.length <= 1}
          >- Remove</button>
        </div>
        {courses.map((c, i) => (
          <div key={i} className="mb-3">
            <input
              className="form-control"
              placeholder={`Course #${i + 1}`}
              value={c.name}
              onChange={ev => handleChange(courses, setCourses, i, 'name', ev.target.value)}
            />
          </div>
        ))}
      </div>

      {/* Certifications */}
      <div className="mb-4">
        <div className="d-flex align-items-center mb-2">
          <label className="form-label mb-0 flex-grow-1">Certifications</label>
          <button
            className="btn btn-sm btn-outline-primary ms-2"
            onClick={() => addEntry(certifications, setCertifications, { name: '', description: '' })}
          >+ Add</button>
          <button
            className="btn btn-sm btn-outline-danger ms-2"
            onClick={() => removeEntry(certifications, setCertifications)}
            disabled={certifications.length <= 1}
          >- Remove</button>
        </div>
        {certifications.map((c, i) => (
          <div key={i} className="mb-3 p-3 border rounded">
            <input
              className="form-control mb-2"
              placeholder="Certification Name"
              value={c.name}
              onChange={ev => handleChange(certifications, setCertifications, i, 'name', ev.target.value)}
            />
            <textarea
              className="form-control"
              rows={2}
              placeholder="Description"
              value={c.description}
              onChange={ev => handleChange(certifications, setCertifications, i, 'description', ev.target.value)}
            />
          </div>
        ))}
      </div>

      {/* Submit */}
      <div className="text-center mt-4">
        <button
          className="btn btn-primary btn-lg px-5 py-2 btn-submit"
          onClick={handleSubmit}
          disabled={loading}
        >
          {loading ? 'Submitting...' : 'See My Match'}
        </button>
      </div>
    </div>
  );
};

export default ResumeInputForm;
